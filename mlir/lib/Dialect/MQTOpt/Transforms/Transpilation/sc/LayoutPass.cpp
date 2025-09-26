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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/RoutingStack.h"

#include <cassert>
#include <cstddef>
#include <deque>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <numeric>
#include <vector>

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
 * @brief A queue of qubit SSA values.
 */
using QubitPool = std::deque<Value>;

/**
 * @brief The necessary datastructures to apply the placement.
 */
struct LayoutContext {
  explicit LayoutContext(Architecture& arch) : arch(&arch) {}

  Architecture* arch;
  RoutingStack<Layout<QubitIndex>> stack{};
  QubitPool pool;
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

  SmallVector<QubitIndex> initialLayout(ctx.arch->nqubits());
  std::iota(initialLayout.begin(), initialLayout.end(), 0);

  ctx.stack.emplace(ctx.arch->nqubits());
  ctx.pool.clear();

  /// Create static / hardware qubits for entry_point functions.
  SmallVector<Value> qubits(ctx.arch->nqubits());
  for (QubitIndex i = 0; i < ctx.arch->nqubits(); ++i) {
    auto qubitOp =
        rewriter.create<QubitOp>(rewriter.getInsertionPoint()->getLoc(), i);
    rewriter.setInsertionPointAfter(qubitOp);
    qubits[i] = qubitOp.getQubit();
  }

  /// Initialize pool with qubits in the initial layout order.
  /// Initialize SSA Value <-> Hardware-Index Mapping.
  for (const auto [programIdx, hardwareIdx] : llvm::enumerate(initialLayout)) {
    const Value q = qubits[hardwareIdx];
    ctx.pool.push_back(q);
    ctx.stack.top().add(programIdx, hardwareIdx, q);
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

/**
 * @brief Retrieve free qubit from pool and replace the allocated qubit with it.
 * Reset the qubit if it has already been allocated before.
 */
WalkResult handleAlloc(AllocQubitOp op, LayoutContext& ctx,
                       PatternRewriter& rewriter) {
  if (ctx.pool.empty()) {
    return op.emitOpError(
        "requires one too many qubits for the targeted architecture");
  }

  /// Retrieve free qubit.
  const Value q = ctx.pool.front();
  ctx.pool.pop_front();

  LLVM_DEBUG({
    const QubitIndex index = ctx.stack.top().lookupHardware(q);
    llvm::dbgs() << "handleAlloc: index= " << index << '\n';
  });

  /// Newly allocated?
  const Operation* defOp = q.getDefiningOp();
  if (defOp != nullptr && isa<QubitOp>(defOp)) {
    rewriter.replaceOp(op, q);
    return WalkResult::advance();
  }

  auto reset = rewriter.create<ResetOp>(op.getLoc(), q);
  rewriter.replaceOp(op, reset);

  /// Update layout.
  ctx.stack.top().remapQubitValue(reset.getInQubit(), reset.getOutQubit());

  return WalkResult::advance();
}

/**
 * @brief Release hardware qubit and erase dealloc operation.
 */
WalkResult handleDealloc(DeallocQubitOp op, LayoutContext& ctx,
                         PatternRewriter& rewriter) {
  ctx.pool.push_back(op.getQubit());
  rewriter.eraseOp(op);
  return WalkResult::advance();
}

/**
 * @brief Update layout.
 */
WalkResult handleUnitary(UnitaryInterface op, LayoutContext& ctx) {
  const std::vector<Value> inQubits = op.getAllInQubits();
  const std::vector<Value> outQubits = op.getAllOutQubits();

  for (const auto [in, out] : llvm::zip(inQubits, outQubits)) {
    ctx.stack.top().remapQubitValue(in, out);
  }

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
                         /// mqtopt Dialect
                         .Case<AllocQubitOp>([&](AllocQubitOp op) {
                           return handleAlloc(op, ctx, rewriter);
                         })
                         .Case<DeallocQubitOp>([&](DeallocQubitOp op) {
                           return handleDealloc(op, ctx, rewriter);
                         })
                         .Case<UnitaryInterface>([&](UnitaryInterface op) {
                           return handleUnitary(op, ctx);
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
