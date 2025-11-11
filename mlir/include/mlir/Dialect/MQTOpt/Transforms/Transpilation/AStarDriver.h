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

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/AStarHeuristicRouter.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/RoutingDriverBase.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include <cstddef>
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

struct Layer {
  /// The program indices of the two-qubit gates within this layer.
  SmallVector<QubitIndexPair, 16> gates;
  /// The first two-qubit op in the layer.
  /// Used as front-anchor to trigger the routing.
  Operation* front{};
  //// @returns true iff the layer contains gates.
  [[nodiscard]] bool empty() const { return gates.empty(); }
};

struct Schedule {
  /// A vector of layers.
  SmallVector<Layer, 0> layers;
  /// The scheduled ops.
  SmallVector<Operation*> ops;
};

class SlidingWindow {
public:
  explicit SlidingWindow(ArrayRef<Layer> layers, const std::size_t nlookahead)
      : layers(layers), nlookahead(nlookahead) {}

  /// @returns the current window of layers.
  [[nodiscard]] ArrayRef<Layer> getCurrent() const {
    if (layers.empty()) {
      return {};
    }

    const auto remaining = layers.size() - offset;
    const auto count = std::min(1 + nlookahead, remaining);
    return layers.slice(offset, count);
  }

  /// Advance to the next window.
  void advance() { ++offset; }

private:
  ArrayRef<Layer> layers;
  std::size_t nlookahead;
  std::size_t offset{};
};

class Scheduler {
  /// The region where the schedule ops reside.
  Region* region;

public:
  explicit Scheduler(Region* region) : region(region) {}

  /**
   * @brief Starting from the given layout, schedule all operations and divide
   * the circuit into parallelly executable layers.
   *
   * It schedules `scf.for` and `scf.if` operations (`RegionBranchOpInterface`)
   * but does not recursively schedule the operation within these ops.
   *
   * @returns the schedule.
   */
  [[nodiscard]] Schedule schedule(Layout layout) {
    SyncMap syncMap;
    Schedule schedule;
    SmallVector<Value> qubits(layout.getHardwareQubits());

    while (!qubits.empty()) {
      Layer layer;
      qubits = advanceQubits(qubits, layer, layout, syncMap, schedule.ops);

      /// Only add non-empty layers to the schedule.
      if (!layer.empty()) {
        schedule.layers.emplace_back(layer);
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "schedule: layers=\n";
      for (const auto [i, layer] : llvm::enumerate(schedule.layers)) {
        llvm::dbgs() << '\t' << i << "= ";
        for (const auto [prog0, prog1] : layer.gates) {
          llvm::dbgs() << "(" << prog0 << "," << prog1 << "), ";
        }
        llvm::dbgs() << '\n';
      }
    });

    return schedule;
  }

private:
  class SyncMap {
    /// Counts the amount of occurrences for a given op.
    DenseMap<Operation*, std::size_t> occurrences;
    /// The operations (value) that depend on the release of another operation
    /// (key). This is used to advance passed `mqtopt.barrier`, `scf.for`, and
    /// `scf.if` ops, which, from a routing perspective, act like identities.
    DenseMap<Operation*, SmallVector<Operation*>> pending;

  public:
    /// Increase the occurrence count of the given op and return true iff the
    /// operation can be scheduled. An operation can be scheduled whenever we've
    /// seen it as many times as it has inputs.
    bool sync(Operation* op) {
      return ++occurrences[op] == op->getNumResults();
    }
    /// Returns a reference to the vector of pending operations for a given op.
    SmallVector<Operation*>& getPending(Operation* op) { return pending[op]; }
  };

  struct OneQubitAdvanceResult {
    /// The advanced qubit value on the wire.
    Value q;
    /// The two-qubit unitary which has q as argument.
    Operation* op{};
  };

  SmallVector<Value> advanceQubits(ArrayRef<Value> qubits, Layer& layer,
                                   Layout& layout, SyncMap& syncMap,
                                   SmallVector<Operation*>& chain) {
    SmallVector<Value> next;

    for (const Value q : qubits) {
      if (q.use_empty()) {
        continue;
      }

      const auto res = advanceQubit(q, chain);

      /// Continue, if no (>=2)-qubit op has been found.
      if (res.op == nullptr) {
        continue;
      }

      /// Otherwise, map the current to the advanced qubit value.
      if (q != res.q) {
        layout.remapQubitValue(q, res.q);
      }

      /// Handle the found (>=2)-qubit op based on its type.
      TypeSwitch<Operation*>(res.op)
          /// MQT
          .Case<BarrierOp>([&](BarrierOp op) {
            for (const auto [in, out] :
                 llvm::zip_equal(op.getInQubits(), op.getOutQubits())) {
              if (in == res.q) {
                layout.remapQubitValue(res.q, out);
                next.append(advanceQubits(ArrayRef(Value(out)), layer, layout,
                                          syncMap, syncMap.getPending(op)));
                return;
              }
            }

            if (syncMap.sync(op)) {
              chain.push_back(op);
              chain.append(syncMap.getPending(op));
            }
          })
          .Case<UnitaryInterface>([&](UnitaryInterface op) {
            if (syncMap.sync(op)) {
              /// Add two-qubit op to schedule.
              chain.push_back(op);

              /// Setup front anchor.
              if (layer.front == nullptr) {
                layer.front = op;
              }

              /// Remap values.
              const auto ins = getIns(op);
              const auto outs = getOuts(op);

              layout.remapQubitValue(ins.first, outs.first);
              layout.remapQubitValue(ins.second, outs.second);

              /// Add gate indices to the current layer.
              layer.gates.emplace_back(layout.lookupProgramIndex(outs.first),
                                       layout.lookupProgramIndex(outs.second));

              next.push_back(outs.first);
              next.push_back(outs.second);
            }
          })
          /// SCF
          .Case<RegionBranchOpInterface>([&](RegionBranchOpInterface op) {
            /// Probably have to remap the layout here.
            const Value out = op->getResult(layout.lookupHardwareIndex(res.q));
            layout.remapQubitValue(res.q, out);
            next.append(advanceQubits(ArrayRef(out), layer, layout, syncMap,
                                      syncMap.getPending(op)));

            if (syncMap.sync(op)) {
              chain.push_back(op);
              chain.append(syncMap.getPending(op));
            }
          })
          .Case<scf::YieldOp>([&](scf::YieldOp op) { /* nothing to do. */ })
          .Default([&]([[maybe_unused]] Operation* op) {
            LLVM_DEBUG({
              llvm::dbgs() << "unknown operation in def-use chain: ";
              op->dump();
            });
            llvm_unreachable("unknown operation in def-use chain");
          });
    }

    return next;
  }

  /**
   * @returns todo
   */
  OneQubitAdvanceResult advanceQubit(Value q, SmallVector<Operation*>& chain) {
    OneQubitAdvanceResult res;
    res.q = q;

    while (true) {
      if (res.q.use_empty()) {
        break;
      }

      Operation* user = getUserInRegion(res.q, region);
      if (user == nullptr) {
        /// Must be a branching op:
        user = res.q.getUsers().begin()->getParentOp();
        assert(isa<scf::IfOp>(user));
      }

      TypeSwitch<Operation*>(user)
          /// MQT
          .Case<UnitaryInterface>([&](UnitaryInterface op) {
            if (op->getNumResults() > 1) {
              res.op = op;
              return;
            }

            res.q = op.getOutQubits().front();
            chain.push_back(user);
          })
          .Case<ResetOp>([&](ResetOp op) {
            res.q = op.getOutQubit();
            chain.push_back(user);
          })
          .Case<MeasureOp>([&](MeasureOp op) {
            res.q = op.getOutQubit();
            chain.push_back(user);
          })
          /// SCF
          .Case<RegionBranchOpInterface>(
              [&](RegionBranchOpInterface op) { res.op = op; })
          .Case<scf::YieldOp>([&](scf::YieldOp op) { res.op = op; })
          .Default([&]([[maybe_unused]] Operation* op) {
            LLVM_DEBUG({
              llvm::dbgs() << "unknown operation in def-use chain: ";
              op->dump();
            });
            llvm_unreachable("unknown operation in def-use chain");
          });

      if (res.op != nullptr) {
        break;
      }
    }

    return res;
  }
};

class AStarDriver final : public RoutingDriverBase {
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
    /// Generate schedule.
    Scheduler scheduler(&region);
    Schedule schedule = scheduler.schedule(layout);

    /// Iterate over schedule in sliding windows of size 1 + nlookahead.
    SlidingWindow window(schedule.layers, nlookahead_);

    Operation* prev{};
    for (Operation* curr : schedule.ops) {

      if (prev != nullptr && prev != curr) {
        rewriter.moveOpAfter(curr, prev);
      }

      const OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(curr);

      const auto res =
          TypeSwitch<Operation*, WalkResult>(curr)
              /// mqtopt Dialect
              .Case<UnitaryInterface>([&](UnitaryInterface op) {
                return handleUnitary(op, layout, window, history, rewriter);
              })
              .Case<ResetOp>(
                  [&](ResetOp op) { return handleReset(op, layout); })
              .Case<MeasureOp>(
                  [&](MeasureOp op) { return handleMeasure(op, layout); })
              /// scf Dialect
              .Case<scf::ForOp>([&](scf::ForOp op) {
                return handleFor(op, layout, rewriter);
              })
              .Case<scf::IfOp>(
                  [&](scf::IfOp op) { return handleIf(op, layout, rewriter); })
              .Case<scf::YieldOp>([&](scf::YieldOp op) {
                return handleYield(op, layout, history, rewriter);
              })
              /// Skip the rest.
              .Default([](auto) { return WalkResult::skip(); });

      if (res.wasInterrupted()) {
        return failure();
      }

      prev = curr;
    }

    return success();
  }

  /**
   * @brief Copy the layout and recursively map the loop body.
   */
  WalkResult handleFor(scf::ForOp op, Layout& layout,
                       PatternRewriter& rewriter) const {
    LLVM_DEBUG(llvm::dbgs() << "handleFor: recurse for loop body\n");

    /// Copy layout.
    Layout forLayout(layout);

    /// Forward out-of-loop and in-loop values.
    const auto initArgs = op.getInitArgs().take_front(arch->nqubits());
    const auto results = op.getResults().take_front(arch->nqubits());
    const auto iterArgs = op.getRegionIterArgs().take_front(arch->nqubits());
    for (const auto [arg, res, iter] : llvm::zip(initArgs, results, iterArgs)) {
      layout.remapQubitValue(arg, res);
      forLayout.remapQubitValue(arg, iter);
    }

    /// Recursively handle loop region.
    SWAPHistory history;
    return rewrite(op.getRegion(), forLayout, history, rewriter);
  }

  /**
   * @brief Copy the layout for each branch and recursively map the branches.
   */
  WalkResult handleIf(scf::IfOp op, Layout& layout,
                      PatternRewriter& rewriter) const {

    /// Recursively handle each branch region.

    LLVM_DEBUG(llvm::dbgs() << "handleIf: recurse for then\n");
    Layout ifLayout(layout);
    SWAPHistory ifHistory;
    const auto ifRes =
        rewrite(op.getThenRegion(), ifLayout, ifHistory, rewriter);
    if (ifRes.failed()) {
      return ifRes;
    }

    LLVM_DEBUG(llvm::dbgs() << "handleIf: recurse for else\n");
    Layout elseLayout(layout);
    SWAPHistory elseHistory;
    const auto elseRes =
        rewrite(op.getElseRegion(), elseLayout, elseHistory, rewriter);
    if (elseRes.failed()) {
      return elseRes;
    }

    /// Forward out-of-if values.
    const auto results = op->getResults().take_front(arch->nqubits());
    for (const auto [hw, res] : llvm::enumerate(results)) {
      const Value q = layout.lookupHardwareValue(hw);
      layout.remapQubitValue(q, res);
    }

    return WalkResult::advance();
  }

  /**
   * @brief Indicates the end of a region defined by a scf op.
   *
   * Restores layout by uncomputation and replaces (invalid) yield.
   *
   * Using uncompute has the advantages of (1) being intuitive and
   * (2) preserving the optimality of the original SWAP sequence.
   * Essentially the better the routing algorithm the better the
   * uncompute. Moreover, this has the nice property that routing
   * a 'for' of 'if' region always requires 2 * #(SWAPs required for region)
   * additional SWAPS.
   */
  WalkResult handleYield(scf::YieldOp op, Layout& layout, SWAPHistory& history,
                         PatternRewriter& rewriter) const {
    LLVM_DEBUG(llvm::dbgs() << "handleYield\n");

    /// Uncompute SWAPs.
    this->insertSWAPs(to_vector(llvm::reverse(history)), layout, op.getLoc(),
                      rewriter);

    return WalkResult::advance();
  }

  /**
   * @brief Ensures the executability of two-qubit gates on the given target
   * architecture by inserting SWAPs.
   */
  WalkResult handleUnitary(UnitaryInterface op, Layout& layout,
                           SlidingWindow& window, SWAPHistory& history,
                           PatternRewriter& rewriter) const {
    LLVM_DEBUG(llvm::dbgs()
               << "handleUnitary: gate= " << op->getName() << '\n');

    /// If this is the first two-qubit op in the layer, route the layer
    /// and remap afterwards.

    const auto curr = window.getCurrent();

    /// Current op is front-anchor: route.
    if (!curr.empty() && op == curr.front().front) {
      route(curr, layout, op.getLoc(), history, rewriter);
      window.advance();
    }

    for (const auto [in, out] :
         llvm::zip(op.getAllInQubits(), op.getAllOutQubits())) {
      layout.remapQubitValue(in, out);
    }

    if (isa<SWAPOp>(op)) {
      const auto outs = getOuts(op);
      layout.swap(outs.first, outs.second);
      history.push_back({layout.lookupHardwareIndex(outs.first),
                         layout.lookupHardwareIndex(outs.second)});
    }

    return WalkResult::advance();
  }

  /**
   * @brief Update layout.
   */
  static WalkResult handleReset(ResetOp op, Layout& layout) {
    layout.remapQubitValue(op.getInQubit(), op.getOutQubit());
    return WalkResult::advance();
  }

  /**
   * @brief Update layout.
   */
  static WalkResult handleMeasure(MeasureOp op, Layout& layout) {
    layout.remapQubitValue(op.getInQubit(), op.getOutQubit());
    return WalkResult::advance();
  }

  /**
   * @brief Use A*-search to make the gates in the front layer (layers.front())
   * executable.
   */
  void route(const ArrayRef<Layer> layers, Layout& layout, Location location,
             SWAPHistory& history, PatternRewriter& rewriter) const {
    /// Find SWAPs.
    SmallVector<ArrayRef<QubitIndexPair>> layerIndices;
    layerIndices.reserve(layers.size());
    for (const auto& layer : layers) {
      layerIndices.push_back(layer.gates);
    }
    const auto swaps = router_.route(layerIndices, layout, *arch);
    /// Append SWAPs to history.
    history.append(swaps);
    /// Insert SWAPs.
    RoutingDriverBase::insertSWAPs(swaps, layout, location, rewriter);
  }

  AStarHeuristicRouter router_;
  std::size_t nlookahead_;
};
} // namespace mqt::ir::opt
