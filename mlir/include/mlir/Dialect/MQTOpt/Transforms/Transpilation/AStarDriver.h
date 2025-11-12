/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/AStarHeuristicRouter.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/RoutingDriverBase.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/Debug.h>
#include <memory>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <utility>

#define DEBUG_TYPE "route-sc"

namespace mqt::ir::opt {

using namespace mlir;

class WireIterator {
public:
  using difference_type = std::ptrdiff_t;
  using value_type = Operation*;

  explicit WireIterator(Value q = nullptr) : q(q) { setNextOp(); }

  Operation* operator*() const { return currOp; }

  WireIterator& operator++() {
    setNextQubit();
    setNextOp();
    return *this;
  }

  void operator++(int) { ++*this; }
  bool operator==(const WireIterator& other) const { return other.q == q; }

private:
  void setNextOp() {
    if (q == nullptr) {
      return;
    }
    if (q.use_empty()) {
      q = nullptr;
      currOp = nullptr;
      return;
    }

    currOp = getUserInRegion(q, q.getParentRegion());
    if (currOp == nullptr) {
      /// Must be a branching op:
      currOp = q.getUsers().begin()->getParentOp();
      assert(isa<scf::IfOp>(currOp));
    }
  }

  void setNextQubit() {
    TypeSwitch<Operation*>(currOp)
        /// MQT
        .Case<UnitaryInterface>([&](UnitaryInterface op) {
          for (const auto& [in, out] :
               llvm::zip_equal(op.getAllInQubits(), op.getAllOutQubits())) {
            if (q == in) {
              q = out;
              return;
            }
          }

          llvm_unreachable("unknown qubit value in def-use chain");
        })
        .Case<ResetOp>([&](ResetOp op) { q = op.getOutQubit(); })
        .Case<MeasureOp>([&](MeasureOp op) { q = op.getOutQubit(); })
        /// SCF
        .Case<scf::ForOp>([&](scf::ForOp op) {
          for (const auto& [in, out] :
               llvm::zip_equal(op.getInitArgs(), op.getResults())) {
            if (q == in) {
              q = out;
              return;
            }
          }

          llvm_unreachable("unknown qubit value in def-use chain");
        })
        .Case<scf::YieldOp>([&](scf::YieldOp op) {
          /// End of region. Invalidate iterator.
          q = nullptr;
          currOp = nullptr;
        })
        .Default([&]([[maybe_unused]] Operation* op) {
          LLVM_DEBUG({
            llvm::dbgs() << "unknown operation in def-use chain: ";
            op->dump();
          });
          llvm_unreachable("unknown operation in def-use chain");
        });
  }

  Value q;
  Operation* currOp{};
};

static_assert(std::input_iterator<WireIterator>);

struct Layer {
  /// All unitary ops contained in this layer.
  SmallVector<Operation*, 0> ops;
  /// The program indices of the gates in this layer.
  SmallVector<QubitIndexPair> gates;
};

class AStarDriver final : public RoutingDriverBase {
  /// A vector of layers.
  using LayerVec = SmallVector<Layer>;
  /// Hold and release map.
  using HoldMap = DenseMap<Operation*, WireIterator>;
  /// A vector of SWAP gate indices.
  using SWAPHistory = SmallVector<QubitIndexPair>;

public:
  AStarDriver(const HeuristicWeights& weights, std::size_t nlookahead,
              std::unique_ptr<Architecture> arch, const Statistics& stats)
      : RoutingDriverBase(std::move(arch), stats), router_(weights),
        nlookahead_(nlookahead) {}

private:
  LogicalResult rewrite(func::FuncOp func, PatternRewriter& rewriter) override {
    LLVM_DEBUG(llvm::dbgs() << "handleFunc: " << func.getSymName() << '\n');

    if (!isEntryPoint(func)) {
      LLVM_DEBUG(llvm::dbgs() << "\tskip non entry\n");
      return success(); // Ignore non entry_point functions for now.
    }

    /// Find all static qubits and initialize layout.
    /// In a circuit diagram this corresponds to finding the very
    /// start of each circuit wire.

    Layout layout(arch->nqubits());
    for_each(func.getOps<QubitOp>(), [&](QubitOp op) {
      const std::size_t index = op.getIndex();
      layout.add(index, index, op.getQubit());
    });

    SWAPHistory history;
    return rewrite(func.getBody(), layout, history, rewriter);
  }

  LogicalResult rewrite(Region& region, Layout& layout, SWAPHistory& history,
                        PatternRewriter& rewriter) const {
    /// Find layers.
    LayerVec layers = schedule(layout);
    /// Route the layers. Might break SSA Dominance.
    route(layout, layers, rewriter);
    /// Repair any SSA Dominance Issues.
    for (Block& block : region.getBlocks()) {
      sortTopologically(&block);
    }
    return success();
  }

  void route(const Layout& layout, LayerVec& layers,
             PatternRewriter& rewriter) const {
    Layout routingLayout(layout);
    LayerVec::iterator end = layers.end();
    for (LayerVec::iterator it = layers.begin(); it != end; ++it) {
      LayerVec::iterator lookaheadIt = std::min(end, it + 1 + nlookahead_);

      auto& front = *it; /// == window.front()
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

      const auto swaps = router_.route(windowLayerGates, routingLayout, *arch);
      /// history.append(swaps);

      rewriter.setInsertionPoint(anchor);
      insertSWAPs(swaps, routingLayout, anchor->getLoc(), rewriter);

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

  static LayerVec schedule(const Layout& layout) {
    LayerVec layers;
    HoldMap onHold;
    Layout schedulingLayout(layout);
    SmallVector<WireIterator> wires(llvm::map_range(
        layout.getHardwareQubits(), [](Value q) { return WireIterator(q); }));

    do {
      const auto [layer, next] =
          collectLayerAndAdvance(wires, schedulingLayout, onHold);

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

  static std::pair<Layer, SmallVector<WireIterator>>
  collectLayerAndAdvance(ArrayRef<WireIterator> wires, Layout& schedulingLayout,
                         DenseMap<Operation*, WireIterator>& onHold) {
    /// The collected layer.
    Layer layer;
    /// A vector of iterators for the next iteration.
    SmallVector<WireIterator> next;
    next.reserve(wires.size());

    for (WireIterator it : wires) {
      while (it != WireIterator()) {
        Operation* op = *it;

        if (!isa<UnitaryInterface>(op)) {
          layer.ops.push_back(op);
          ++it;
          continue;
        }

        auto u = cast<UnitaryInterface>(op);
        if (!isTwoQubitGate(u)) {
          /// Add unitary to layer operations.
          layer.ops.push_back(op);
          /// Forward scheduling layout.
          schedulingLayout.remapQubitValue(u.getInQubits().front(),
                                           u.getOutQubits().front());
          ++it;
          continue;
        }

        if (onHold.contains(u)) {
          const auto ins = getIns(u);
          const auto outs = getOuts(u);

          /// Release iterators for next iteration.
          next.push_back(onHold.lookup(u));
          next.push_back(++it);
          /// Only add ready two-qubit gates to the layer.
          layer.ops.push_back(op);
          layer.gates.emplace_back(
              schedulingLayout.lookupProgramIndex(ins.first),
              schedulingLayout.lookupProgramIndex(ins.second));
          /// Forward scheduling layout.
          schedulingLayout.remapQubitValue(ins.first, outs.first);
          schedulingLayout.remapQubitValue(ins.second, outs.second);
        } else {
          /// Emplace the next iterator after the two-qubit
          /// gate for a later release.
          onHold.try_emplace(u, ++it);
        }

        break;
      }
    }

    return {layer, next};
  }

  AStarHeuristicRouter router_;
  std::size_t nlookahead_;
};
} // namespace mqt::ir::opt
