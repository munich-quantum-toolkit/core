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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/AStarHeuristicRouter.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/WireIterator.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include <cassert>
#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <optional>
#include <utility>

#define DEBUG_TYPE "route-astar-sc"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ASTARROUTINGPASSSC
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

/// @brief A vector of SWAP gate indices.
using SWAPHistory = SmallVector<QubitIndexPair>;

/// @brief A composite datastructure for LLVM Statistics.
struct Statistics {
  llvm::Statistic* numSwaps;
};

/// @brief A composite datastructure for pass parameters.
struct Params {
  /// @brief The amount of lookahead layers.
  std::size_t nlookahead;
  float alpha;
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

struct Layer {
  /// @brief All unitary ops contained in this layer.
  SmallVector<Operation*, 0> ops;
  /// @brief The program indices of the gates in this layer.
  SmallVector<QubitIndexPair> gates;
};

/// @brief A vector of layers.
using LayerVec = SmallVector<Layer>;

/// @brief Map to handle multi-qubit gates when traversing the def-use chain.
class SynchronizationMap {
  DenseMap<Operation*, SmallVector<WireIterator, 0>> onHold;
  DenseMap<Operation*, std::size_t> refCount;

public:
  /// @returns true iff. the operation is contained in the map.
  bool contains(Operation* op) { return onHold.contains(op); }
  /// @brief Add op with respective iterator and ref count to map.
  void add(Operation* op, WireIterator it, const std::size_t cnt) {
    onHold.try_emplace(op, SmallVector<WireIterator>{it});
    /// Decrease the cnt by one because the op was visited when adding.
    refCount.try_emplace(op, cnt - 1);
  }

  std::optional<SmallVector<WireIterator, 0>> visit(Operation* op,
                                                    WireIterator it) {
    assert(refCount.contains(op) && "expected sync map to contain op");

    /// Add iterator for later release.
    onHold[op].push_back(it);

    /// Release iterators whenever the ref count reaches zero.
    if (--refCount[op] == 0) {
      return onHold[op];
    }

    return std::nullopt;
  }
};

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
          "route: swap= p%d:h%d, p%d:h%d <- p%d:h%d, p%d:h%d\n", prog1, hw0,
          prog0, hw1, prog0, hw0, prog1, hw1);
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

std::pair<Layer, SmallVector<WireIterator>>
collectLayerAndAdvance(ArrayRef<WireIterator> wires, Layout& schedulingLayout,
                       SynchronizationMap& sync) {
  /// The collected layer.
  Layer layer;
  /// A vector of iterators for the next iteration.
  SmallVector<WireIterator> next;
  next.reserve(wires.size());

  for (WireIterator it : wires) {
    while (it != WireIterator()) {
      Operation* op = *it;

      /// A barrier may be a UnitaryInterface, but also requires
      /// synchronization.
      if (auto barrier = dyn_cast<BarrierOp>(op)) {
        if (!sync.contains(barrier)) {
          sync.add(barrier, ++it, barrier.getInQubits().size());
          break;
        }

        if (const auto iterators = sync.visit(barrier, ++it)) {
          layer.ops.push_back(barrier);
          next.append(iterators.value());
        }

        break;
      }

      if (auto u = dyn_cast<UnitaryInterface>(op)) {
        if (!isTwoQubitGate(u)) {
          /// Add unitary to layer operations.
          layer.ops.push_back(op);

          /// Forward scheduling layout.
          schedulingLayout.remapQubitValue(u.getInQubits().front(),
                                           u.getOutQubits().front());
          ++it;
          continue;
        }

        if (!sync.contains(u)) {
          /// Add the next iterator after the two-qubit
          /// gate for a later release.
          sync.add(u, ++it, 2);
          break;
        }

        if (const auto iterators = sync.visit(u, ++it)) {
          const auto ins = getIns(u);
          const auto outs = getOuts(u);

          /// Release iterators for next iteration.
          next.append(iterators.value());

          /// Only add ready two-qubit gates to the layer.
          layer.ops.push_back(u);
          layer.gates.emplace_back(
              schedulingLayout.lookupProgramIndex(ins.first),
              schedulingLayout.lookupProgramIndex(ins.second));

          /// Forward scheduling layout.
          schedulingLayout.remapQubitValue(ins.first, outs.first);
          schedulingLayout.remapQubitValue(ins.second, outs.second);
        }

        break;
      }

      /// RegionBranchOpInterface = scf.for, scf.if, ...
      if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
        if (!sync.contains(branch)) {
          /// This assumes that branching ops always take and return all
          /// hardware qubits.
          sync.add(branch, ++it, schedulingLayout.getNumQubits());
          break;
        }

        if (const auto iterators = sync.visit(branch, ++it)) {
          layer.ops.push_back(branch);
          next.append(iterators.value());
        }

        break;
      }

      /// Anything else is either a measure or reset op.
      assert(isa<MeasureOp>(op) ||
             isa<ResetOp>(op) && "expect measure or reset");
      layer.ops.push_back(op);
      ++it;
    }
  }

  return {layer, next};
}

LayerVec schedule(const Layout& layout) {
  LayerVec layers;
  SynchronizationMap sync;
  Layout schedulingLayout(layout);
  SmallVector<WireIterator> wires(llvm::map_range(
      layout.getHardwareQubits(), [](Value q) { return WireIterator(q); }));

  do {
    const auto [layer, next] =
        collectLayerAndAdvance(wires, schedulingLayout, sync);

    /// Early exit if there are no more gates to route.
    if (layer.gates.empty()) {
      break;
    }

    layers.emplace_back(layer);
    wires = next;

  } while (!wires.empty());

  LLVM_DEBUG({
    llvm::dbgs() << "schedule: layers=\n";
    for (const auto [i, layer] : llvm::enumerate(layers)) {
      llvm::dbgs() << '\t' << i << "= ";
      for (const auto [prog0, prog1] : layer.gates) {
        llvm::dbgs() << "(" << prog0 << "," << prog1 << "), ";
      }
      llvm::dbgs() << '\n';
    }
  });

  return layers;
}

void routeEachLayer(LayerVec& layers, const Layout& layout,
                    RoutingContext& ctx) {
  Layout routingLayout(layout);
  LayerVec::iterator end = layers.end();
  for (LayerVec::iterator it = layers.begin(); it != end; ++it) {
    LayerVec::iterator lookaheadIt = std::min(end, it + 1 + ctx.nlookahead);

    auto& front = *it; // == window.front()
    auto window = llvm::make_range(it, lookaheadIt);
    auto windowLayerGates = to_vector(llvm::map_range(
        window, [](const Layer& layer) { return ArrayRef(layer.gates); }));

    Operation* anchor{}; /// First op in textual IR order.
    for (Operation* op : front.ops) {
      if (anchor == nullptr || op->isBeforeInBlock(anchor)) {
        anchor = op;
      }
    }

    assert(anchor != nullptr && "expected to find anchor");
    llvm::dbgs() << "schedule: anchor= " << *anchor << '\n';

    const auto swaps =
        ctx.router.route(windowLayerGates, routingLayout, *ctx.arch);
    /// history.append(swaps);

    ctx.rewriter.setInsertionPoint(anchor);
    insertSWAPs(anchor->getLoc(), swaps, routingLayout, ctx.rewriter);

    for (Operation* op : front.ops) {
      if (auto u = dyn_cast<UnitaryInterface>(op)) {
        for (const auto& [in, out] :
             llvm::zip_equal(u.getAllInQubits(), u.getAllOutQubits())) {
          routingLayout.remapQubitValue(in, out);
        }
        continue;
      }
      llvm_unreachable("TODO.");
    }
  }
}

LogicalResult processRegion(Region& region, Layout& layout,
                            RoutingContext& ctx) {
  /// Find layers.
  LayerVec layers = schedule(layout);
  /// Route each of the layers. Might violate SSA dominance.
  routeEachLayer(layers, layout, ctx);
  /// Repair any SSA dominance issues.
  for (Block& block : region.getBlocks()) {
    sortTopologically(&block);
  }
  return success();
}

LogicalResult processFunction(func::FuncOp func, RoutingContext& ctx) {
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

  return processRegion(func.getBody(), layout, ctx);
}

/**
 * @brief Route the given module for the targeted architecture using A*-search.
 *
 * @param module The module to route.
 * @param arch The targeted architecture.
 * @param stats The composite statistics datastructure.
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
    if (processFunction(func, ctx).failed()) {
      return failure();
    }
  }
  return success();
}

/**
 * @brief This pass ensures that the connectivity constraints of the target
 * architecture are met.
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
