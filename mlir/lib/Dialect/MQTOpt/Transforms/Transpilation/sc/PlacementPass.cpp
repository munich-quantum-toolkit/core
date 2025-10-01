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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Stack.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <deque>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <numeric>
#include <random>

#define DEBUG_TYPE "placement-sc"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_PLACEMENTPASSSC
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

/**
 * @brief A queue of hardware indices.
 */
using HardwareIndexPool = std::deque<QubitIndex>;

/**
 * @brief A base class for all initial placement strategies.
 */
class InitialPlacer {
public:
  InitialPlacer() = default;
  InitialPlacer(const InitialPlacer&) = default;
  InitialPlacer& operator=(const InitialPlacer&) = default;
  InitialPlacer(InitialPlacer&&) noexcept = default;
  InitialPlacer& operator=(InitialPlacer&&) noexcept = default;
  virtual ~InitialPlacer() = default;
  [[nodiscard]] virtual SmallVector<QubitIndex> operator()() = 0;
};

/**
 * @brief Identity initial placement.
 */
class IdentityPlacer final : public InitialPlacer {
public:
  explicit IdentityPlacer(const std::size_t nqubits) : nqubits_(nqubits) {}

  [[nodiscard]] SmallVector<QubitIndex> operator()() override {
    SmallVector<QubitIndex> mapping(nqubits_);
    std::iota(mapping.begin(), mapping.end(), 0);
    return mapping;
  }

private:
  std::size_t nqubits_;
};

/**
 * @brief Random initial placement.
 */
class RandomPlacer final : public InitialPlacer {
public:
  explicit RandomPlacer(const std::size_t nqubits, const std::mt19937_64& rng)
      : nqubits_(nqubits), rng_(rng) {}

  [[nodiscard]] SmallVector<QubitIndex> operator()() override {
    SmallVector<QubitIndex> mapping(nqubits_);
    std::iota(mapping.begin(), mapping.end(), 0);
    std::ranges::shuffle(mapping, rng_);
    return mapping;
  }

private:
  std::size_t nqubits_;
  std::mt19937_64 rng_;
};

/**
 * @brief The necessary datastructures to apply the placement.
 */
struct PlacementContext {
  explicit PlacementContext(Architecture& arch, InitialPlacer& placer)
      : arch(&arch), placer(&placer) {}

  Architecture* arch;
  InitialPlacer* placer;
  HardwareIndexPool pool;
  LayoutStack<Layout<QubitIndex>> stack{};
};

/**
 * @brief Adds the necessary hardware qubits for entry_point functions and
 * prepares the stack for the qubit placement in the function body.
 */
WalkResult handleFunc(func::FuncOp op, PlacementContext& ctx,
                      PatternRewriter& rewriter) {
  assert(ctx.stack.empty() && "handleFunc: stack must be empty");

  rewriter.setInsertionPointToStart(&op.getBody().front());
  LLVM_DEBUG({
    llvm::dbgs() << "handleFunc: entry_point= " << op.getSymName() << '\n';
  });

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
  for (const auto layout = (*ctx.placer)();
       const auto [programIdx, hardwareIdx] : llvm::enumerate(layout)) {
    const Value q = qubits[hardwareIdx];
    ctx.pool.push_back(hardwareIdx);
    ctx.stack.top().add(programIdx, hardwareIdx, q);
  }

  return WalkResult::advance();
}

/**
 * @brief Indicates the end of a region defined by a function. Consequently, pop
 * the region's state from the stack.
 */
WalkResult handleReturn(PlacementContext& ctx) {
  ctx.stack.pop();
  return WalkResult::advance();
}

/**
 * @brief Replaces the 'for' loop with one that has all hardware qubits as init
 * arguments.
 *
 * Prepares the stack for the placement of the loop body by adding a copy of the
 * current state to the stack. Forwards the results in the parent state.
 */
WalkResult handleFor(scf::ForOp op, PlacementContext& ctx,
                     PatternRewriter& rewriter) {
  const std::size_t nargs = op.getBody()->getNumArguments();
  const std::size_t nresults = op->getNumResults();

  /// Construct new init arguments.
  const ArrayRef<Value> qubits = ctx.stack.top().getHardwareQubits();
  const SmallVector<Value> newInitArgs(qubits);

  /// Replace old for with a new one with updated init arguments.
  auto forOp = rewriter.create<scf::ForOp>(op.getLoc(), op.getLowerBound(),
                                           op.getUpperBound(), op.getStep(),
                                           newInitArgs);

  rewriter.mergeBlocks(op.getBody(), forOp.getBody(),
                       forOp.getBody()->getArguments().take_front(nargs));

  if (nresults > 0) {
    rewriter.replaceOp(op, forOp.getResults().take_front(nresults));
  } else {
    rewriter.eraseOp(op);
  }

  /// Prepare stack.
  ctx.stack.duplicateTop();

  // Forward out-of-loop and in-loop state.
  for (const auto [arg, res, iter] :
       llvm::zip(forOp.getInitArgs(), forOp.getResults(),
                 forOp.getRegionIterArgs())) {
    if (isa<QubitType>(arg.getType())) {
      ctx.stack.getItemAtDepth(FOR_PARENT_DEPTH).remapQubitValue(arg, res);
      ctx.stack.top().remapQubitValue(arg, iter);
    }
  }

  return WalkResult::advance();
}

/**
 * @brief Replaces the 'if' statement with one that has all hardware qubits as
 * result.
 *
 * Prepares the stack for the placement of the 'then' and 'else' body by adding
 * a copy of the current state to the stack for each branch. Forwards the
 * results in the parent state.
 */
WalkResult handleIf(scf::IfOp op, PlacementContext& ctx,
                    PatternRewriter& rewriter) {
  const std::size_t nresults = op->getNumResults();

  /// Construct new result types.
  const ArrayRef<Value> qubits = ctx.stack.top().getHardwareQubits();
  const auto rng = llvm::map_range(qubits, [](Value q) { return q.getType(); });
  const SmallVector<Type> resultTypes(rng.begin(), rng.end());

  /// Replace old if with a new one with updated result types.
  const bool hasElse = !op.getElseRegion().empty();
  auto ifOp = rewriter.create<scf::IfOp>(op.getLoc(), resultTypes,
                                         op.getCondition(), hasElse);

  rewriter.mergeBlocks(&op.getThenRegion().front(),
                       &ifOp.getThenRegion().front());
  if (hasElse) {
    rewriter.mergeBlocks(&op.getElseRegion().front(),
                         &ifOp.getElseRegion().front());
  }

  if (nresults > 0) {
    rewriter.replaceOp(op, ifOp.getResults().take_front(nresults));
  } else {
    rewriter.eraseOp(op);
  }

  /// Prepare stack.
  ctx.stack.duplicateTop(); // Else
  ctx.stack.duplicateTop(); // Then

  /// Forward results for all hardware qubits.
  Layout<QubitIndex>& stateBeforeIf = ctx.stack.getItemAtDepth(IF_PARENT_DEPTH);
  for (std::size_t i = 0; i < qubits.size(); ++i) {
    const Value in = stateBeforeIf.getHardwareQubits()[i];
    const Value out = ifOp->getResult(i);
    stateBeforeIf.remapQubitValue(in, out);
  }

  return WalkResult::advance();
}

/**
 * @brief Indicates the end of a region defined by a branching op. Consequently,
 * we pop the region's state from the stack.
 */
WalkResult handleYield(scf::YieldOp op, PlacementContext& ctx,
                       PatternRewriter& rewriter) {
  if (!isa<scf::ForOp>(op->getParentOp()) &&
      !isa<scf::IfOp>(op->getParentOp())) {
    return WalkResult::skip();
  }

  rewriter.replaceOpWithNewOp<scf::YieldOp>(
      op, ctx.stack.top().getHardwareQubits());

  ctx.stack.pop();

  return WalkResult::advance();
}

/**
 * @brief Retrieve free qubit from pool and replace the allocated qubit with it.
 * Reset the qubit if it has already been allocated before.
 */
WalkResult handleAlloc(AllocQubitOp op, PlacementContext& ctx,
                       PatternRewriter& rewriter) {
  if (ctx.pool.empty()) {
    return op.emitOpError(
        "requires one too many qubits for the targeted architecture");
  }

  /// Retrieve free qubit.
  const std::size_t index = ctx.pool.front();
  ctx.pool.pop_front();

  LLVM_DEBUG({ llvm::dbgs() << "handleAlloc: index= " << index << '\n'; });

  const Value q = ctx.stack.top().lookupHardware(index);

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
WalkResult handleDealloc(DeallocQubitOp op, PlacementContext& ctx,
                         PatternRewriter& rewriter) {
  const std::size_t index = ctx.stack.top().lookupHardware(op.getQubit());
  ctx.pool.push_back(index);
  rewriter.eraseOp(op);
  return WalkResult::advance();
}

/**
 * @brief Update layout.
 */
WalkResult handleReset(ResetOp op, PlacementContext& ctx) {
  ctx.stack.top().remapQubitValue(op.getInQubit(), op.getOutQubit());
  return WalkResult::advance();
}

/**
 * @brief Update layout.
 */
WalkResult handleMeasure(MeasureOp op, PlacementContext& ctx) {
  ctx.stack.top().remapQubitValue(op.getInQubit(), op.getOutQubit());
  return WalkResult::advance();
}

/**
 * @brief Update layout.
 */
WalkResult handleUnitary(UnitaryInterface op, PlacementContext& ctx) {
  for (const auto [in, out] :
       llvm::zip(op.getAllInQubits(), op.getAllOutQubits())) {
    ctx.stack.top().remapQubitValue(in, out);
  }

  return WalkResult::advance();
}

LogicalResult run(ModuleOp module, MLIRContext* mlirCtx,
                  PlacementContext& ctx) {
  PatternRewriter rewriter(mlirCtx);

  /// Prepare work-list.
  SmallVector<Operation*> worklist;
  for (const auto func : module.getOps<func::FuncOp>()) {
    if (!isEntryPoint(func)) {
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

    const auto res =
        TypeSwitch<Operation*, WalkResult>(curr)
            /// built-in Dialect
            .Case<ModuleOp>(
                [&](ModuleOp /* op */) { return WalkResult::advance(); })
            /// func Dialect
            .Case<func::FuncOp>(
                [&](func::FuncOp op) { return handleFunc(op, ctx, rewriter); })
            .Case<func::ReturnOp>(
                [&](func::ReturnOp /* op */) { return handleReturn(ctx); })
            /// scf Dialect
            .Case<scf::ForOp>(
                [&](scf::ForOp op) { return handleFor(op, ctx, rewriter); })
            .Case<scf::IfOp>(
                [&](scf::IfOp op) { return handleIf(op, ctx, rewriter); })
            .Case<scf::YieldOp>(
                [&](scf::YieldOp op) { return handleYield(op, ctx, rewriter); })
            /// mqtopt Dialect
            .Case<AllocQubitOp>(
                [&](AllocQubitOp op) { return handleAlloc(op, ctx, rewriter); })
            .Case<DeallocQubitOp>([&](DeallocQubitOp op) {
              return handleDealloc(op, ctx, rewriter);
            })
            .Case<ResetOp>([&](ResetOp op) { return handleReset(op, ctx); })
            .Case<MeasureOp>(
                [&](MeasureOp op) { return handleMeasure(op, ctx); })
            .Case<UnitaryInterface>(
                [&](UnitaryInterface op) { return handleUnitary(op, ctx); })
            /// Skip the rest.
            .Default([](auto) { return WalkResult::skip(); });

    if (res.wasInterrupted()) {
      return failure();
    }
  }

  return success();
}

/**
 * @brief This pass maps dynamic qubits to static qubits on superconducting
 * quantum devices using initial placement strategies.
 */
struct PlacementPassSC final : impl::PlacementPassSCBase<PlacementPassSC> {
  using PlacementPassSCBase::PlacementPassSCBase;

  void runOnOperation() override {
    const auto arch = getArchitecture(ArchitectureName::MQTTest);
    const auto placer = getPlacer(*arch);

    if (PlacementContext ctx(*arch, *placer);
        failed(run(getOperation(), &getContext(), ctx))) {
      signalPassFailure();
    }
  }

private:
  [[nodiscard]] std::unique_ptr<InitialPlacer>
  getPlacer(const Architecture& arch) const {
    switch (static_cast<PlacementStrategy>(strategy)) {
    case PlacementStrategy::Identity:
      LLVM_DEBUG({ llvm::dbgs() << "getPlacer: identity placement\n"; });
      return std::make_unique<IdentityPlacer>(arch.nqubits());
    case PlacementStrategy::Random:
      std::random_device rd;
      const std::size_t seed = rd();
      LLVM_DEBUG({
        llvm::dbgs() << "getPlacer: random placement with seed = " << seed
                     << '\n';
      });
      return std::make_unique<RandomPlacer>(arch.nqubits(),
                                            std::mt19937_64(seed));
    }
    llvm_unreachable("Unknown strategy");
  }
};
} // namespace
} // namespace mqt::ir::opt
