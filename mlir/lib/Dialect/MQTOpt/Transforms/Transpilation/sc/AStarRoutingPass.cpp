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

struct Wire {
  Wire(WireIterator it, QubitIndex index) : it(it), index(index) {}
  WireIterator it;
  QubitIndex index;
};

/// @brief Map to handle multi-qubit gates when traversing the def-use chain.
class SynchronizationMap {
  /// @brief Maps operations to to-be-released iterators.
  DenseMap<Operation*, SmallVector<Wire, 0>> onHold;

  /// @brief Maps operations to ref counts. An op can be released whenever the
  /// count reaches zero.
  DenseMap<Operation*, std::size_t> refCount;

public:
  /// @returns true iff. the operation is contained in the map.
  bool contains(Operation* op) { return onHold.contains(op); }

  /// @brief Add op with respective wire and ref count to map.
  void add(Operation* op, Wire wire, const std::size_t cnt) {
    onHold.try_emplace(op, SmallVector<Wire>{wire});
    /// Decrease the cnt by one because the op was visited when adding.
    refCount.try_emplace(op, cnt - 1);
  }

  /// @brief Decrement ref count of op and potentially release its iterators.
  std::optional<SmallVector<Wire, 0>> visit(Operation* op, Wire wire) {
    assert(refCount.contains(op) && "expected sync map to contain op");

    /// Add iterator for later release.
    onHold[op].push_back(wire);

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

//===----------------------------------------------------------------------===//
// Scheduling
//===----------------------------------------------------------------------===//

SmallVector<Wire, 2> advanceUntilEndOfTwoQubitBlock(ArrayRef<Wire> wires,
                                                    Layer& layer) {
  assert(wires.size() == 2 && "expected two wires");

  WireIterator end;
  auto [it0, index0] = wires[0];
  auto [it1, index1] = wires[1];
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
      layer.addOp(u0); // Add 1Q-op to layer operations.
      ++it0;
      continue;
    }

    UnitaryInterface u1 = cast<UnitaryInterface>(op1);

    /// Advance for single qubit gate on wire 1.
    if (!isTwoQubitGate(u1)) {
      layer.addOp(u1); // Add 1Q-op to layer operations.
      ++it1;
      continue;
    }

    /// Stop if the wires reach different two qubit gates.
    if (op0 != op1) {
      break;
    }

    /// Remap and advance if u0 == u1.
    layer.addOp(u1);

    ++it0;
    ++it1;
  }

  return {Wire(it0, index0), Wire(it1, index1)};
}

/**
 * @brief Advance each wire until (>=2)-qubit gates are found, collect the
 * indices of the respective two-qubit gates, and prepare iterators for next
 * iteration.
 */
std::pair<Layer, SmallVector<Wire>>
collectLayerAndAdvance(ArrayRef<Wire> wires, SynchronizationMap& sync,
                       const std::size_t nqubits) {
  /// The collected layer.
  Layer layer;
  /// A vector of iterators for the next iteration.
  SmallVector<Wire> next;
  next.reserve(wires.size());

  for (auto [it, index] : wires) {
    while (it != WireIterator()) {
      Operation* curr = *it;

      /// A barrier may be a UnitaryInterface, but also requires
      /// synchronization.
      if (auto op = dyn_cast<BarrierOp>(curr)) {
        if (!sync.contains(op)) {
          sync.add(op, Wire(++it, index), op.getInQubits().size());
          break;
        }

        if (const auto iterators = sync.visit(op, Wire(++it, index))) {
          layer.addOp(op);
          next.append(iterators.value());
        }

        break;
      }

      if (auto op = dyn_cast<UnitaryInterface>(curr)) {
        if (!isTwoQubitGate(op)) {
          layer.addOp(curr); // Add 1Q-op to layer operations.
          ++it;
          continue;
        }

        if (!sync.contains(op)) {
          /// Add the next iterator after the two-qubit
          /// gate for a later release.
          sync.add(op, Wire(++it, index), 2);
          break;
        }

        if (const auto iterators = sync.visit(op, Wire(++it, index))) {
          /// Only add ready two-qubit gates to the layer.
          layer.addOp(op);
          layer.gates.emplace_back((*iterators)[0].index,
                                   (*iterators)[1].index);

          /// Release iterators for next iteration.
          next.append(advanceUntilEndOfTwoQubitBlock(iterators.value(), layer));
        }

        break;
      }

      if (auto op = dyn_cast<ResetOp>(curr)) {
        layer.addOp(curr);
        ++it;
        continue;
      }

      if (auto op = dyn_cast<MeasureOp>(curr)) {
        layer.addOp(curr);
        ++it;
        continue;
      }

      if (auto op = dyn_cast<RegionBranchOpInterface>(curr)) {
        if (!sync.contains(op)) {
          /// This assumes that branch ops always returns all hardware qubits.
          sync.add(op, Wire(++it, index), nqubits);
          break;
        }

        if (const auto iterators = sync.visit(op, Wire(++it, index))) {
          layer.addOp(op);
          next.append(iterators.value());
        }
        break;
      }

      if (auto yield = dyn_cast<scf::YieldOp>(curr)) {
        if (!sync.contains(yield)) {
          /// This assumes that yield always returns all hardware qubits.
          sync.add(yield, Wire(++it, index), nqubits);
          break;
        }

        if (const auto iterators = sync.visit(yield, Wire(++it, index))) {
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
LayerVec schedule(const Layout& layout, Region& region) {
  LayerVec layers;
  SynchronizationMap sync;
  SmallVector<Wire> wires;
  wires.reserve(layout.getNumQubits());
  for (auto [hw, q] : llvm::enumerate(layout.getHardwareQubits())) {
    wires.emplace_back(WireIterator(q, &region), layout.getProgramIndex(hw));
  }

  do {
    const auto [layer, next] =
        collectLayerAndAdvance(wires, sync, layout.getNumQubits());
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

//===----------------------------------------------------------------------===//
// Routing
//===----------------------------------------------------------------------===//

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
 *
 * If A*-search fails use the LightSABRE fallback mechanism: Search
 * the the gate which acts on the qubit index pair with the minimum distance in
 * the coupling graph of the targeted architecture. Route this gate naively,
 * remove it from the front layer, and restart the A*-search.
 */
void findAndInsertSWAPs(ArrayRef<ArrayRef<QubitIndexPair>> layers,
                        Location location, Layout& layout,
                        SmallVector<QubitIndexPair>& history,
                        RoutingContext& ctx) {
  /// Mutable copy of the front layer.
  SmallVector<QubitIndexPair> workingFront(layers.front());

  SmallVector<ArrayRef<QubitIndexPair>> workingLayers;
  workingLayers.reserve(layers.size());
  workingLayers.push_back(workingFront); // Non-owning view of workingFront.
  for (auto layer : layers.drop_front()) {
    workingLayers.push_back(layer);
  }

  while (!workingFront.empty()) {
    if (const auto swaps = ctx.router.route(workingLayers, layout, *ctx.arch)) {
      history.append(*swaps);
      insertSWAPs(location, *swaps, layout, ctx.rewriter);
      *(ctx.stats.numSwaps) += swaps->size();
      return;
    }

    QubitIndexPair bestGate;
    std::size_t bestIdx = 0;
    std::size_t minDist = std::numeric_limits<std::size_t>::max();
    for (const auto [i, gate] : llvm::enumerate(workingFront)) {
      const auto hw0 = layout.getHardwareIndex(gate.first);
      const auto hw1 = layout.getHardwareIndex(gate.second);
      const auto dist = ctx.arch->distanceBetween(hw0, hw1);
      if (dist < minDist) {
        bestIdx = i;
        minDist = dist;
        bestGate = std::make_pair(hw0, hw1);
      }
    }

    workingFront.erase(workingFront.begin() + static_cast<ptrdiff_t>(bestIdx));

    SmallVector<QubitIndexPair, 16> swaps;
    const auto path =
        ctx.arch->shortestPathBetween(bestGate.first, bestGate.second);
    for (std::size_t i = 0; i < path.size() - 2; ++i) {
      swaps.emplace_back(path[i], path[i + 1]);
    }

    history.append(swaps);
    insertSWAPs(location, swaps, layout, ctx.rewriter);
    *(ctx.stats.numSwaps) += swaps.size();
  }
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

      findAndInsertSWAPs(windowLayerGates, front.anchor->getLoc(), layout,
                         history, ctx);
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
  for (Block& block : region.getBlocks()) {
    sortTopologically(&block);
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
