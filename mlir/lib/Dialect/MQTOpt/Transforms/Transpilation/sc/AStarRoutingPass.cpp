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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
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
#include <optional>
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

struct Layer {
  /// @brief All ops contained in this layer.
  SmallVector<Operation*, 0> ops;
  /// @brief The program indices of the gates in this layer.
  SmallVector<QubitIndexPair> gates;
  /// @brief The first op in ops in textual IR order.
  Operation* anchor{};
  /// @brief Add op to ops and reset anchor if necessary.
  void addOp(Operation* op) {
    ops.emplace_back(op);
    if (anchor == nullptr || op->isBeforeInBlock(anchor)) {
      anchor = op;
    }
  }
  /// @returns true iff the layer contains gates to route.
  [[nodiscard]] bool hasRoutableGates() const { return !gates.empty(); }
};

/// @brief A vector of layers.
using LayerVec = SmallVector<Layer>;

/// @brief Map to handle multi-qubit gates when traversing the def-use chain.
class SynchronizationMap {
  /// @brief Maps operations to to-be-released iterators.
  DenseMap<Operation*, SmallVector<WireIterator, 0>> onHold;

  /// @brief Maps operations to ref counts. An op can be released whenever the
  /// count reaches zero.
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

  /// @brief Decrement ref count of op and potentially release its iterators.
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

SmallVector<WireIterator, 2>
advanceUntilEndOfTwoQubitBlock(ArrayRef<WireIterator> wires, Layout& layout,
                               Layer& layer) {
  assert(wires.size() == 2 && "expected two wires");

  WireIterator it0 = wires[0];
  WireIterator it1 = wires[1];
  WireIterator end;
  while (it0 != end && it1 != end) {
    Operation* op0 = *it0;
    if (!isa<UnitaryInterface>(op0) || isa<BarrierOp>(op0)) {
      break;
    }

    Operation* op1 = *it1;
    if (!isa<UnitaryInterface>(op1) || isa<BarrierOp>(op1)) {
      break;
    }

    UnitaryInterface u0 = cast<UnitaryInterface>(op0);

    /// Advance for single qubit gate on wire 0.
    if (!isTwoQubitGate(u0)) {
      layer.addOp(u0);   // Add 1Q-op to layer operations.
      remap(u0, layout); // Remap scheduling layout.
      ++it0;
      continue;
    }

    UnitaryInterface u1 = cast<UnitaryInterface>(op1);

    /// Advance for single qubit gate on wire 1.
    if (!isTwoQubitGate(u1)) {
      layer.addOp(u1);   // Add 1Q-op to layer operations.
      remap(u1, layout); // Remap scheduling layout.
      ++it1;
      continue;
    }

    /// Stop if the wires reach different two qubit gates.
    if (op0 != op1) {
      break;
    }

    /// Remap and advance if u0 == u1.
    layer.addOp(u1);
    remap(u1, layout);

    ++it0;
    ++it1;
  }

  return {it0, it1};
}

/**
 * @brief Advance each wire until (>=2)-qubit gates are found, collect the
 * indices of the respective two-qubit gates, and prepare iterators for next
 * iteration.
 */
std::pair<Layer, SmallVector<WireIterator>>
collectLayerAndAdvance(ArrayRef<WireIterator> wires, Layout& layout,
                       SynchronizationMap& sync) {
  /// The collected layer.
  Layer layer;
  /// A vector of iterators for the next iteration.
  SmallVector<WireIterator> next;
  next.reserve(wires.size());

  for (WireIterator it : wires) {
    while (it != WireIterator()) {
      Operation* curr = *it;

      /// A barrier may be a UnitaryInterface, but also requires
      /// synchronization.
      if (auto op = dyn_cast<BarrierOp>(curr)) {
        if (!sync.contains(op)) {
          sync.add(op, ++it, op.getInQubits().size());
          break;
        }

        if (const auto iterators = sync.visit(op, ++it)) {
          layer.addOp(op);
          next.append(iterators.value());
          remap(op, layout); // Remap values on release.
        }

        break;
      }

      if (auto op = dyn_cast<UnitaryInterface>(curr)) {
        if (!isTwoQubitGate(op)) {
          layer.addOp(curr); // Add 1Q-op to layer operations.
          remap(op, layout); // Remap scheduling layout.
          ++it;
          continue;
        }

        if (!sync.contains(op)) {
          /// Add the next iterator after the two-qubit
          /// gate for a later release.
          sync.add(op, ++it, 2);
          break;
        }

        if (const auto iterators = sync.visit(op, ++it)) {
          const auto ins = getIns(op);

          /// Only add ready two-qubit gates to the layer.
          layer.addOp(op);
          layer.gates.emplace_back(layout.lookupProgramIndex(ins.first),
                                   layout.lookupProgramIndex(ins.second));
          remap(op, layout); // Remap scheduling layout.

          /// Release iterators for next iteration.
          next.append(
              advanceUntilEndOfTwoQubitBlock(iterators.value(), layout, layer));
        }

        break;
      }

      if (auto op = dyn_cast<ResetOp>(curr)) {
        remap(op, layout);
        layer.addOp(curr);
        ++it;
        continue;
      }

      if (auto op = dyn_cast<MeasureOp>(curr)) {
        remap(op, layout);
        layer.addOp(curr);
        ++it;
        continue;
      }

      if (auto op = dyn_cast<RegionBranchOpInterface>(curr)) {
        if (!sync.contains(op)) {
          /// This assumes that branch ops always returns all hardware qubits.
          sync.add(op, ++it, layout.getNumQubits());
          break;
        }

        if (const auto iterators = sync.visit(op, ++it)) {
          layer.addOp(op);

          /// Remap values on release.
          for (const auto& [in, out] :
               llvm::zip_equal(layout.getHardwareQubits(), op->getResults())) {
            layout.remapQubitValue(in, out);
          }
          next.append(iterators.value());
        }
        break;
      }

      if (auto yield = dyn_cast<scf::YieldOp>(curr)) {
        if (!sync.contains(yield)) {
          /// This assumes that yield always returns all hardware qubits.
          sync.add(yield, ++it, layout.getNumQubits());
          break;
        }

        if (const auto iterators = sync.visit(yield, ++it)) {
          layer.addOp(yield);
        }

        break;
      }

      /// Anything else is a bug in the program.
      assert(false && "unhandled operation");
    }
  }

  return {layer, next};
}

/**
 * @brief Given a layout, divide the circuit into layers and schedule the ops in
 * their respective layer.
 */
LayerVec schedule(Layout layout, Region& region) {
  LayerVec layers;
  SynchronizationMap sync;
  SmallVector<WireIterator> wires(
      llvm::map_range(layout.getHardwareQubits(),
                      [&](Value q) { return WireIterator(q, &region); }));

  do {
    const auto [layer, next] = collectLayerAndAdvance(wires, layout, sync);
    if (layer.ops.empty()) {
      break;
    }
    layers.emplace_back(layer);
    wires = next;
  } while (!wires.empty());

  LLVM_DEBUG({
    llvm::dbgs() << "schedule: layers=\n";
    for (const auto [i, layer] : llvm::enumerate(layers)) {
      llvm::dbgs() << '\t' << i << ": ";
      llvm::dbgs() << "#ops= " << layer.ops.size() << ", ";
      llvm::dbgs() << "gates= ";
      if (layer.hasRoutableGates()) {
        for (const auto [prog0, prog1] : layer.gates) {
          llvm::dbgs() << "(" << prog0 << "," << prog1 << "), ";
        }
      } else {
        llvm::dbgs() << "(), ";
      }
      if (layer.anchor) {
        llvm::dbgs() << "anchor= " << layer.anchor->getLoc() << ", ";
      }
      llvm::dbgs() << '\n';
    }
  });

  return layers;
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
 * @brief Route each layer by iterating a sliding window of (1 + nlookahead)
 * layers.
 */
LogicalResult routeEachLayer(const LayerVec& layers, Layout layout,
                             SmallVector<QubitIndexPair>& history,
                             RoutingContext& ctx) {
  const auto nhorizion = static_cast<std::ptrdiff_t>(1 + ctx.nlookahead);

  LayerVec::const_iterator end = layers.end();
  LayerVec::const_iterator it = layers.begin();
  for (; it != end; std::advance(it, 1)) {
    LayerVec::const_iterator lookaheadIt =
        std::min(end, std::next(it, nhorizion));

    const auto& front = *it; // == window.front()

    if (front.hasRoutableGates()) {
      ctx.rewriter.setInsertionPoint(front.anchor);

      /// Find SWAPs for front layer with nlookahead layers.
      const auto window = llvm::make_range(it, lookaheadIt);
      const auto windowLayerGates = to_vector(llvm::map_range(
          window, [](const Layer& layer) { return ArrayRef(layer.gates); }));
      const auto swaps = ctx.router.route(windowLayerGates, layout, *ctx.arch);

      /// Append SWAPs to history.
      history.append(swaps);

      /// Insert SWAPs.
      insertSWAPs(front.anchor->getLoc(), swaps, layout, ctx.rewriter);

      /// Count SWAPs.
      *(ctx.stats.numSwaps) += swaps.size();
    }

    for (const Operation* curr : front.ops) {
      const auto res = TypeSwitch<const Operation*, WalkResult>(curr)
                           .Case<UnitaryInterface>([&](UnitaryInterface op) {
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
 * @brief Schedule and route the given region.
 *
 * Since this might break SSA Dominance, sort the blocks in the given region
 * topologically.
 */
LogicalResult processRegion(Region& region, Layout& layout,
                            SmallVector<QubitIndexPair>& history,
                            RoutingContext& ctx) {
  /// Find and route each of the layers. Might violate SSA dominance.
  const auto res =
      routeEachLayer(schedule(layout, region), layout, history, ctx);
  if (res.failed()) {
    return res;
  }

  /// Repair any SSA dominance issues.
  // for (Block& block : region.getBlocks()) {
  //   sortTopologically(&block);
  // }

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
    SmallVector<Wire> circuit;
    circuit.reserve(ctx.arch->nqubits());
    for_each(func.getOps<QubitOp>(), [&](QubitOp op) {
      const std::size_t index = op.getIndex();
      layout.add(index, index, op.getQubit());
      circuit.emplace_back(op.getQubit(), op.getQubit());
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
