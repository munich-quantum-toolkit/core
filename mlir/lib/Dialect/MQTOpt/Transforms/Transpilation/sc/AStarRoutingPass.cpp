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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Router.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Schedule.h"

#include <cassert>
#include <cstddef>
#include <limits>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/LogicalResult.h>
#include <memory>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <utility>

#define DEBUG_TYPE "route-astar-sc"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ASTARROUTINGPASSSC
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

/// @brief A composite datastructure for LLVM Statistics.
struct Statistics {
  llvm::Statistic* numSwaps;
};

/// @brief A composite datastructure for pass parameters.
struct Params {
  /// @brief The amount of lookahead layers.
  std::size_t nlookahead;
  /// @brief The alpha factor in the heuristic function.
  float alpha;
  /// @brief The lambda decay factor in the heuristic function.
  float lambda;
};

/// @brief Commonly passed parameters for the routing functions.
struct RoutingContext {
  /// @brief The targeted architecture.
  std::unique_ptr<Architecture> arch;
  /// @brief LLVM/MLIR statistics.
  Statistics stats;
  /// @brief A pattern rewriter.
  PatternRewriter rewriter;
  /// @brief The A*-search based router.
  AStarHeuristicRouter router;
  /// @brief The amount of lookahead layers.
  std::size_t nlookahead;
};

LogicalResult processRegion(Region& region, Layout& layout,
                            SmallVector<QubitIndexPair>& history,
                            RoutingContext& ctx);

/**
 * @brief Insert SWAP ops at the rewriter's insertion point.
 *
 * @param location The location of the inserted SWAP ops.
 * @param swaps The hardware indices of the SWAPs.
 * @param layout The current layout.
 * @param rewriter The pattern rewriter.
 */
void insertSWAPs(Location location, ArrayRef<QubitIndexPair> swaps,
                 Layout& layout, PatternRewriter& rewriter) {
  for (const auto [hw0, hw1] : swaps) {
    const Value in0 = layout.lookupHardwareValue(hw0);
    const Value in1 = layout.lookupHardwareValue(hw1);
    [[maybe_unused]] const auto [prog0, prog1] =
        layout.getProgramIndices(hw0, hw1);

    LLVM_DEBUG({
      llvm::dbgs() << llvm::format(
          "insertSWAPs: swap= p%d:h%d, p%d:h%d <- p%d:h%d, p%d:h%d\n", prog1,
          hw0, prog0, hw1, prog0, hw0, prog1, hw1);
    });

    auto swap = createSwap(location, in0, in1, rewriter);
    const auto [out0, out1] = getOuts(swap);

    rewriter.setInsertionPointAfter(swap);
    replaceAllUsesInRegionAndChildrenExcept(in0, out1, swap->getParentRegion(),
                                            swap, rewriter);
    replaceAllUsesInRegionAndChildrenExcept(in1, out0, swap->getParentRegion(),
                                            swap, rewriter);

    layout.swap(in0, in1);
    layout.remapQubitValue(in0, out0);
    layout.remapQubitValue(in1, out1);
  }
}

/**
 * @brief Copy the layout and recursively process the loop body.
 */
WalkResult handle(scf::ForOp op, Layout& layout, RoutingContext& ctx) {
  /// Copy layout.
  Layout forLayout(layout);

  /// Forward out-of-loop and in-loop values.
  const auto initArgs = op.getInitArgs().take_front(ctx.arch->nqubits());
  const auto results = op.getResults().take_front(ctx.arch->nqubits());
  const auto iterArgs = op.getRegionIterArgs().take_front(ctx.arch->nqubits());
  for (const auto [arg, res, iter] : llvm::zip(initArgs, results, iterArgs)) {
    layout.remapQubitValue(arg, res);
    forLayout.remapQubitValue(arg, iter);
  }

  /// Recursively handle loop region.
  SmallVector<QubitIndexPair> history;
  return processRegion(op.getRegion(), forLayout, history, ctx);
}

/**
 * @brief Copy the layout for each branch and recursively process the branches.
 */
WalkResult handle(scf::IfOp op, Layout& layout, RoutingContext& ctx) {
  /// Recursively handle each branch region.
  Layout ifLayout(layout);
  SmallVector<QubitIndexPair> ifHistory;

  const auto ifRes =
      processRegion(op.getThenRegion(), ifLayout, ifHistory, ctx);
  if (ifRes.failed()) {
    return ifRes;
  }

  Layout elseLayout(layout);
  SmallVector<QubitIndexPair> elseHistory;
  const auto elseRes =
      processRegion(op.getElseRegion(), elseLayout, elseHistory, ctx);
  if (elseRes.failed()) {
    return elseRes;
  }

  /// Forward out-of-if values.
  const auto results = op->getResults().take_front(ctx.arch->nqubits());
  for (const auto [in, out] : llvm::zip(layout.getHardwareQubits(), results)) {
    layout.remapQubitValue(in, out);
  }

  return WalkResult::advance();
}

/**
 * @brief Indicates the end of a region defined by a scf op.
 *
 * Restores layout by uncomputation.
 */
WalkResult handle(scf::YieldOp op, Layout& layout,
                  ArrayRef<QubitIndexPair> history, RoutingContext& ctx) {
  /// Uncompute SWAPs.
  insertSWAPs(op.getLoc(), llvm::to_vector(llvm::reverse(history)), layout,
              ctx.rewriter);
  /// Count SWAPs.
  *(ctx.stats.numSwaps) += history.size();
  return WalkResult::advance();
}

/**
 * @brief Remap all input to output qubits for the given unitary op. SWAP
 * indices if the unitary is a SWAP.
 */
WalkResult handle(UnitaryInterface op, Layout& layout,
                  SmallVector<QubitIndexPair>& history) {
  remap(op, layout);
  if (isa<SWAPOp>(op)) {
    const auto outs = getOuts(op);
    layout.swap(outs.first, outs.second);
    history.push_back({layout.lookupHardwareIndex(outs.first),
                       layout.lookupHardwareIndex(outs.second)});
  }
  return WalkResult::advance();
}

/**
 * @brief Find and insert SWAPs using A*-search.
 */
void findAndInsertSWAPs(MutableArrayRef<Schedule::GateLayer> window,
                        Operation* anchor, Layout& layout,
                        SmallVector<QubitIndexPair>& history,
                        RoutingContext& ctx) {
  ctx.rewriter.setInsertionPoint(anchor);

  if (const auto swaps = ctx.router.route(window, layout, *ctx.arch)) {
    if (!swaps->empty()) {
      history.append(*swaps);
      insertSWAPs(anchor->getLoc(), *swaps, layout, ctx.rewriter);
      *(ctx.stats.numSwaps) += swaps->size();
    }
    return;
  }

  throw std::runtime_error("A* router failed to find a valid SWAP sequence");
}

/**
 * @brief Schedule and route the given region.
 *
 * Since this might break SSA Dominance, sort the blocks in the given region
 * topologically.
 */
LogicalResult processRegion(Region& region, Layout& layout,
                            SmallVector<QubitIndexPair>& history,
                            RoutingContext& ctx) {
  /// Find and route each of the layers. Might violate SSA dominance.
  auto schedule = getSchedule(layout, region);

  for (std::size_t i = 0; i < schedule.gateLayers.size(); ++i) {
    auto window = schedule.getWindow(i, ctx.nlookahead);
    auto opLayer = schedule.opLayers[i];

    findAndInsertSWAPs(window, opLayer.anchor, layout, history, ctx);

    for (Operation* curr : opLayer.ops) {
      ctx.rewriter.setInsertionPoint(curr);
      const auto res = TypeSwitch<Operation*, WalkResult>(curr)
                           .Case<UnitaryInterface>([&](UnitaryInterface op) {
                             if (i + 1 < schedule.gateLayers.size()) {
                               ctx.rewriter.moveOpBefore(
                                   op, schedule.opLayers[i + 1].anchor);
                             }
                             return handle(op, layout, history);
                           })
                           .Case<ResetOp>([&](ResetOp op) {
                             remap(op, layout);
                             return WalkResult::advance();
                           })
                           .Case<MeasureOp>([&](MeasureOp op) {
                             remap(op, layout);
                             return WalkResult::advance();
                           })
                           .Case<scf::ForOp>([&](scf::ForOp op) {
                             return handle(op, layout, ctx);
                           })
                           .Case<scf::IfOp>([&](scf::IfOp op) {
                             return handle(op, layout, ctx);
                           })
                           .Case<scf::YieldOp>([&](scf::YieldOp op) {
                             return handle(op, layout, history, ctx);
                           })
                           .Default([](auto) { return WalkResult::skip(); });
      if (res.wasInterrupted()) {
        return failure();
      }
    }
  }

  return success();
}

/**
 * @brief Route the given module for the targeted architecture using A*-search.
 * Processes each entry_point function separately.
 */
LogicalResult route(ModuleOp module, std::unique_ptr<Architecture> arch,
                    Params& params, Statistics& stats) {
  const HeuristicWeights weights(params.alpha, params.lambda,
                                 params.nlookahead);
  RoutingContext ctx{.arch = std::move(arch),
                     .stats = stats,
                     .rewriter = PatternRewriter(module->getContext()),
                     .router = AStarHeuristicRouter(weights),
                     .nlookahead = params.nlookahead};
  for (auto func : module.getOps<func::FuncOp>()) {
    LLVM_DEBUG(llvm::dbgs() << "handleFunc: " << func.getSymName() << '\n');

    if (!isEntryPoint(func)) {
      LLVM_DEBUG(llvm::dbgs() << "\tskip non entry\n");
      return success(); // Ignore non entry_point functions for now.
    }

    /// Find all hardware (static) qubits and initialize layout.
    Layout layout(ctx.arch->nqubits());
    for_each(func.getOps<QubitOp>(), [&](QubitOp op) {
      const std::size_t index = op.getIndex();
      layout.add(index, index, op.getQubit());
    });

    SmallVector<QubitIndexPair> history;
    const auto res = processRegion(func.getBody(), layout, history, ctx);
    if (res.failed()) {
      return res;
    }
  }
  return success();
}

/**
 * @brief Routes the program by dividing the circuit into layers of parallel
 * two-qubit gates and iteratively searches and inserts SWAPs for each layer
 * using A*-search.
 */
struct AStarRoutingPassSC final
    : impl::AStarRoutingPassSCBase<AStarRoutingPassSC> {
  using AStarRoutingPassSCBase<AStarRoutingPassSC>::AStarRoutingPassSCBase;

  void runOnOperation() override {
    if (preflight().failed()) {
      signalPassFailure();
      return;
    }

    auto arch = getArchitecture(archName);
    if (!arch) {
      emitError(UnknownLoc::get(&getContext()))
          << "unsupported architecture '" << archName << "'";
      signalPassFailure();
      return;
    }

    Statistics stats{.numSwaps = &numSwaps};
    Params params{.nlookahead = nlookahead, .alpha = alpha, .lambda = lambda};
    if (route(getOperation(), std::move(arch), params, stats).failed()) {
      signalPassFailure();
    };
  }

private:
  LogicalResult preflight() {
    if (archName.empty()) {
      return emitError(UnknownLoc::get(&getContext()),
                       "required option 'arch' not provided");
    }

    return success();
  }
};

} // namespace
} // namespace mqt::ir::opt
