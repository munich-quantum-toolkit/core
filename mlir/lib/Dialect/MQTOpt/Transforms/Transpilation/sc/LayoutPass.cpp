/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/RoutingStack.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/LogicalResult.h"

#include <cassert>
#include <cstddef>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#define DEBUG_TYPE "layout-sc"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_LAYOUTPASSSC
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

/**
 * @brief A function attribute that specifies an (QIR) entry point function.
 */
constexpr llvm::StringLiteral ENTRY_POINT_ATTR{"entry_point"};

/**
 * @brief The datatype for qubit indices. For now, 64bit.
 */
using QubitIndex = std::size_t;

/**
 * @brief Maps SSA values to hardware indices.
 */
using QubitIndexMap = llvm::DenseMap<Value, QubitIndex>;

/**
 * @brief Replace an old SSA value with a new one.
 */
void remapQubitValue(QubitIndexMap& state, const Value in, const Value out) {
  assert(in != out && "'in' must not equal 'out'");
  const auto it = state.find(in);
  assert(it != state.end() && "map must contain 'in'");
  state[out] = it->second;
  state.erase(in);
}

/**
 * @brief Manages free / used hardware indices.
 */
struct HardwareIndexPool {

  /**
   * @brief Fill the pool with indices determined by the given layout.
   */
  void fill(ArrayRef<QubitIndex> layout) {
    freeHardwareIndices_.clear();
    for (const QubitIndex i : llvm::reverse(layout)) {
      freeHardwareIndices_.insert(i);
    }
  }

  /**
   * @brief Re-insert hardware index to set of free indices.
   */
  void release(const QubitIndex index) { freeHardwareIndices_.insert(index); }

  /**
   * @brief Retrieve free hardware index if available.
   * @returns The index, or std::nullopt if none is available.
   */
  [[nodiscard]] std::optional<QubitIndex> retrieve() {
    if (freeHardwareIndices_.empty()) {
      return std::nullopt;
    }
    const QubitIndex index = freeHardwareIndices_.back();
    freeHardwareIndices_.pop_back();
    return index;
  }

  /**
   * @brief Return true if the index is in-use, i.e., it has been allocated
   * before.
   */
  [[nodiscard]] bool isUsed(const QubitIndex index) const {
    return !freeHardwareIndices_.contains(index);
  }

private:
  /**
   * @brief Set of free hardware indices.
   *
   * The SetVector ensures a deterministic iteration order.
   */
  llvm::SetVector<QubitIndex> freeHardwareIndices_;
};

template <class Context> class TranspilationStep {
public:
  explicit TranspilationStep<Context>(Context& cntx, MLIRContext* ctx)
      : cntx_(cntx), rewriter_(ctx) {}

  virtual ~TranspilationStep() = default;
  virtual WalkResult handleFunc(func::FuncOp op) {}
  virtual WalkResult handleReturn(func::ReturnOp op) {}
  virtual WalkResult handleAlloc(AllocQubitOp op) {}
  virtual WalkResult handleDealloc(DeallocQubitOp op) {}

  LogicalResult run(ModuleOp module) {

    /// Prepare work-list.
    SmallVector<Operation*> worklist;
    for (const auto func : module.getOps<func::FuncOp>()) {
      if (!func->hasAttr(ENTRY_POINT_ATTR)) {
        continue; // Ignore non entry_point functions for now.
      }
      func->walk<WalkOrder::PreOrder>(
          [&](Operation* op) { worklist.push_back(op); });
    }

    /// Iterate work-list.
    for (Operation* curr : worklist) {
      if (curr == nullptr) {
        continue; // Skip erased ops.
      }

      const OpBuilder::InsertionGuard guard(rewriter_);
      rewriter_.setInsertionPoint(curr);

      const auto res =
          TypeSwitch<Operation*, WalkResult>(curr)
              /// built-in Dialect
              .template Case<ModuleOp>(
                  [&](ModuleOp /* op */) { return WalkResult::advance(); })
              /// func Dialect
              .template Case<func::FuncOp>(
                  [&](func::FuncOp op) { return handleFunc(op); })
              .template Case<func::ReturnOp>(
                  [&](func::ReturnOp op) { return handleReturn(op); })
              /// mqtopt Dialect
              // .Case<AllocQubitOp>([&](AllocQubitOp op) {
              //   return handleAlloc(op, ctx.stack(), ctx.pool(), rewriter);
              // })
              // .Case<DeallocQubitOp>([&](DeallocQubitOp op) {
              //   return handleDealloc(op, ctx.stack(), ctx.pool(), rewriter);
              // })
              /// Skip the rest.
              .Default([](auto) { return WalkResult::skip(); });

      if (res.wasInterrupted()) {
        return failure();
      }
    }
  }

protected:
  Context cntx_;
  PatternRewriter rewriter_;
};

struct LayoutContext {
  explicit LayoutContext(Architecture& arch) : arch(&arch) {}

  Architecture* arch;
  RoutingStack<QubitIndexMap> stack{};
  HardwareIndexPool pool{};
};

/**
 * @brief Adds nqubit 'mqtopt.qubit' ops for entry_point functions.
 * Initializes the pool and hence applies the initial layout. Consequently,
 * the initial layout is always applied at the beginning of a function. Pushes
 * newly-initialized state on stack.
 */
WalkResult handleFunc(func::FuncOp op, LayoutContext& ctx,
                      PatternRewriter& rewriter) {
  assert(ctx.stack.empty() && "handleFunc: stack must be empty");

  rewriter.setInsertionPointToStart(&op.getBody().front());
  LLVM_DEBUG({
    llvm::dbgs() << "handleFunc: entry_point= " << op.getSymName() << '\n';
  });

  //   ctx.ilg().generate();

  //   LLVM_DEBUG({
  //     llvm::dbgs() << "handleFunc: initial layout= ";
  //     ctx.ilg().dump();
  //     llvm::dbgs() << '\n';
  //   });

  ctx.stack.emplace();

  for (std::size_t i = 0; i < ctx.arch->nqubits(); ++i) {
    auto qubitOp =
        rewriter.create<QubitOp>(rewriter.getInsertionPoint()->getLoc(), i);
    rewriter.setInsertionPointAfter(qubitOp);

    ctx.stack.top().try_emplace(qubitOp.getQubit(), i);
  }

  return WalkResult::advance();
}

/**
 * @brief Defines the end of a region (and hence routing state) defined by a
 * function. Consequently, we pop the region's state from the stack. Since
 * we currently only route entry_point functions we do not need to return
 * all static qubits here.
 */
WalkResult handleReturn(LayoutContext& ctx) {
  ctx.stack.pop();
  return WalkResult::advance();
}

LogicalResult run(ModuleOp module, MLIRContext* mlirCtx, LayoutContext& ctx) {
  PatternRewriter rewriter(mlirCtx);

  /// Prepare work-list.
  SmallVector<Operation*> worklist;
  for (const auto func : module.getOps<func::FuncOp>()) {
    if (!func->hasAttr(ENTRY_POINT_ATTR)) {
      continue; // Ignore non entry_point functions for now.
    }
    func->walk<WalkOrder::PreOrder>(
        [&](Operation* op) { worklist.push_back(op); });
  }

  /// Iterate work-list.
  for (Operation* curr : worklist) {
    if (curr == nullptr) {
      continue; // Skip erased ops.
    }

    const OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(curr);

    const auto res = TypeSwitch<Operation*, WalkResult>(curr)
                         /// built-in Dialect
                         .Case<ModuleOp>([&](ModuleOp /* op */) {
                           return WalkResult::advance();
                         })
                         /// func Dialect
                         .Case<func::FuncOp>([&](func::FuncOp op) {
                           return handleFunc(op, ctx, rewriter);
                         })
                         .Case<func::ReturnOp>([&](func::ReturnOp /* op */) {
                           return handleReturn(ctx);
                         })
                         /// Skip the rest.
                         .Default([](auto) { return WalkResult::skip(); });

    if (res.wasInterrupted()) {
      return failure();
    }
  }

  return success();
}

/**
 * @brief TODO
 */
struct LayoutPassSC final : impl::LayoutPassSCBase<LayoutPassSC> {
  void runOnOperation() override {
    auto arch = getArchitecture(ArchitectureName::MQTTest);

    LayoutContext ctx(*arch);
    if (failed(run(getOperation(), &getContext(), ctx))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mqt::ir::opt
