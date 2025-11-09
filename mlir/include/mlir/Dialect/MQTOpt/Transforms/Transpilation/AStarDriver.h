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
  SmallVector<QubitIndexPair> twoQubitIndices;

  /// Operations that can be executed before the two-qubit gates.
  SmallVector<Operation*> ops;
  /// The two-qubit gates.
  SmallVector<Operation*> twoQubitOps;
  /// Operations that will be executable whenever all gates in the layer are
  /// executable.
  SmallVector<Operation*> blockOps;

  /// @returns the scheduled operations in the layer.
  [[nodiscard]] auto getOps() {
    return concat<Operation*>(ops, twoQubitOps, blockOps);
  }
};

using Schedule = SmallVector<Layer, 0>;

class Scheduler {
  /// The region where the schedule ops reside.
  Region* region;
  /// Counts the amount of occurrences for a given op.
  /// Currently only used for two-qubit unitaries and scf ops.
  DenseMap<Operation*, std::size_t> occurrences;

public:
  explicit Scheduler(Region* region) : region(region) {}

  /**
   * @brief Starting from the given layout, schedule all operations and divide
   * the circuit into parallelly executable layers.
   * @returns the schedule.
   */
  [[nodiscard]] Schedule schedule(Layout layout) {
    occurrences.clear();

    Schedule schedule;
    SmallVector<Value> qubits(layout.getHardwareQubits()); // worklist.
    while (!qubits.empty()) {
      const auto res = getNextLayer(qubits, layout);
      schedule.emplace_back(res.layer);
      qubits = res.qubits;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "schedule: layers=\n";
      for (const auto [i, layer] : llvm::enumerate(schedule)) {
        llvm::dbgs() << '\t' << i << "= ";
        for (const auto [prog0, prog1] : layer.twoQubitIndices) {
          llvm::dbgs() << "(" << prog0 << "," << prog1 << "), ";
        }
        llvm::dbgs() << '\n';
      }
    });

    return schedule;
  }

private:
  struct NextLayerResult {
    /// The next layer.
    Layer layer;
    /// The updated worklist of qubits for the next iteration.
    SmallVector<Value> qubits;
  };

  struct AdvanceResult {
    /// The advanced qubit value on the wire.
    Value q = nullptr;
    /// The (>=2)-qubit op which has q as argument.
    Operation* op = nullptr;
    /// All one-qubit ops visited until the (>=2)-qubit op.
    SmallVector<Operation*> ops;
    /// @returns true iff the advancement found a (>=2)-qubit op.
    explicit operator bool() const { return op != nullptr; }
  };

  struct BlockAdvanceResult {
    /// The final qubit values in the two-qubit block.
    ValuePair outs;
    /// All operations visited in the two-qubit block.
    SmallVector<Operation*> ops;
  };

  /**
   * @returns todo
   */
  [[nodiscard]] NextLayerResult getNextLayer(ArrayRef<const Value> qubits,
                                             Layout& layout) {
    NextLayerResult next;
    next.qubits.reserve(qubits.size());

    for (const Value q : qubits) {
      if (q.use_empty()) {
        continue;
      }

      const auto res = advanceOneQubitOnWire(q);

      /// Append one-qubit op chain to layer.
      next.layer.ops.append(res.ops);

      /// If no (>=2)-qubit op has been found continue.
      if (!res) {
        continue;
      }

      /// Otherwise map the current to the advanced qubit value.
      if (q != res.q) {
        layout.remapQubitValue(q, res.q);
      }

      /// Handle the found (>=2)-qubit op based on its type:

      if (auto barrier = dyn_cast<BarrierOp>(res.op)) {
        /// Once we've seen all inputs of the barrier, we can release and
        /// forwards its qubit values.
        if (++occurrences[barrier] == barrier->getNumResults()) {
          for (const auto [in, out] :
               llvm::zip_equal(barrier.getInQubits(), barrier.getOutQubits())) {
            layout.remapQubitValue(in, out);
          }
          next.layer.blockOps.push_back(barrier);
        }
        continue;
      }

      if (auto u = dyn_cast<UnitaryInterface>(res.op)) {
        if (++occurrences[u] == 2) {
          const auto ins = getIns(u);
          const auto outs = getOuts(u);

          next.layer.twoQubitOps.emplace_back(u);
          next.layer.twoQubitIndices.emplace_back(
              layout.lookupProgramIndex(ins.first),
              layout.lookupProgramIndex(ins.second));

          layout.remapQubitValue(ins.first, outs.first);
          layout.remapQubitValue(ins.second, outs.second);

          next.qubits.push_back(outs.first);
          next.qubits.push_back(outs.second);

          // const auto blockResult = skipTwoQubitBlock(outs, layout);
          // next.layer.blockOps.append(blockResult.ops);

          // if (blockResult.outs.first != nullptr &&
          //     !blockResult.outs.first.use_empty()) {
          //   next.qubits.push_back(blockResult.outs.first);
          // }

          // if (blockResult.outs.second != nullptr &&
          //     !blockResult.outs.second.use_empty()) {
          //   next.qubits.push_back(blockResult.outs.second);
          // }
        }
        continue;
      }

      if (auto loop = dyn_cast<scf::ForOp>(res.op)) {
        if (++occurrences[loop] == loop->getNumResults()) {
          for (const auto [in, out] :
               llvm::zip_equal(loop.getInitArgs(), loop.getResults())) {
            layout.remapQubitValue(in, out);
            next.qubits.push_back(out);
          }
          next.layer.blockOps.push_back(loop);
        }
        continue;
      }

      if (auto cond = dyn_cast<scf::IfOp>(res.op)) {
        if (++occurrences[cond] == cond->getNumResults()) {
          for (const auto [in, out] :
               llvm::zip_equal(layout.getHardwareQubits(), cond.getResults())) {
            layout.remapQubitValue(in, out);
            next.qubits.push_back(out);
          }
          next.layer.blockOps.push_back(cond);
        }
      }

      if (auto yield = dyn_cast<scf::YieldOp>(res.op)) {
        if (++occurrences[yield] == yield.getResults().size()) {
          next.layer.blockOps.push_back(yield);
        }
      }
    }

    return next;
  }

  /**
   * @returns todo
   */
  AdvanceResult advanceOneQubitOnWire(const Value q) {
    AdvanceResult res;
    res.q = q;

    while (true) {
      if (res.q.use_empty()) {
        break;
      }

      Operation* user = getUserInRegion(res.q, region);
      if (user == nullptr) {
        /// Must be a branching op:
        user = res.q.getUsers().begin()->getParentOp();
        assert(isa<RegionBranchOpInterface>(user));
      }

      TypeSwitch<Operation*>(user)
          /// MQT
          .Case<BarrierOp>([&](BarrierOp op) { res.op = op; })
          .Case<UnitaryInterface>([&](UnitaryInterface op) {
            if (isTwoQubitGate(op)) {
              res.op = op;
              return; // Found a two-qubit gate, stop advancing head.
            }
            // Otherwise, advance head.
            res.q = op.getOutQubits().front();
            res.ops.push_back(user); /// Only add one-qubit gates.
          })
          .Case<ResetOp>([&](ResetOp op) {
            res.q = op.getOutQubit();
            res.ops.push_back(user);
          })
          .Case<MeasureOp>([&](MeasureOp op) {
            res.q = op.getOutQubit();
            res.ops.push_back(user);
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

  /**
   * @returns todo
   */
  BlockAdvanceResult skipTwoQubitBlock(const ValuePair outs, Layout& layout) {
    BlockAdvanceResult blockResult;

    std::array<UnitaryInterface, 2> gates;
    std::array<Value, 2> heads{outs.first, outs.second};

    while (true) {
      bool stop = false;
      for (const auto [i, q] : llvm::enumerate(heads)) {
        const auto res = advanceOneQubitOnWire(q);
        blockResult.ops.append(res.ops);
        if (!res) {
          heads[i] = nullptr;
          stop = true;
          break;
        }

        if (q != res.q) {
          layout.remapQubitValue(q, res.q);
        }

        heads[i] = res.q;
        gates[i] = dyn_cast<UnitaryInterface>(res.op);
      }

      if (stop || gates[0] != gates[1]) {
        break;
      }

      blockResult.ops.push_back(gates[0]);

      const ValuePair ins = getIns(gates[0]);
      const ValuePair outs = getOuts(gates[0]);
      layout.remapQubitValue(ins.first, outs.first);
      layout.remapQubitValue(ins.second, outs.second);
      heads = {outs.first, outs.second};
    }

    blockResult.outs = std::make_pair(heads[0], heads[1]);

    return blockResult;
  }
};

class AStarDriver final : public RoutingDriverBase {
  using SWAPHistory = SmallVector<QubitIndexPair>;

public:
  AStarDriver(std::unique_ptr<Architecture> arch,
              const HeuristicWeights& weights, std::size_t nlookahead)
      : RoutingDriverBase(std::move(arch)), router_(weights),
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

    /// Generate schedule for region.

    Scheduler scheduler(&region);
    Schedule schedule = scheduler.schedule(layout);

    /// Iterate over schedule in sliding windows of size 1 + nlookahead.

    Operation* prev{};

    for (Schedule::iterator it = schedule.begin(); it != schedule.end(); ++it) {
      const Schedule::iterator end =
          std::min(it + 1 + nlookahead_, schedule.end());
      const auto window = llvm::make_range(it, end);
      const auto ops = window.begin()->getOps();

      for (Operation* curr : ops) {

        /// Impose a strict ordering of the operations. Note that
        /// we reorder the operation before we insert any swaps.

        if (prev != nullptr && prev != curr && curr->isBeforeInBlock(prev)) {
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
                .Case<scf::IfOp>([&](scf::IfOp op) {
                  return handleIf(op, layout, rewriter);
                })
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
  static WalkResult handleYield(scf::YieldOp op, Layout& layout,
                                SWAPHistory& history,
                                PatternRewriter& rewriter) {
    LLVM_DEBUG(llvm::dbgs() << "handleYield\n");

    /// Uncompute SWAPs.
    RoutingDriverBase::insertSWAPs(llvm::to_vector(llvm::reverse(history)),
                                   layout, op.getLoc(), rewriter);

    return WalkResult::advance();
  }

  /**
   * @brief Ensures the executability of two-qubit gates on the given target
   * architecture by inserting SWAPs.
   */
  WalkResult handleUnitary(UnitaryInterface op, Layout& layout,
                           const ArrayRef<const Layer> window,
                           SWAPHistory& history,
                           PatternRewriter& rewriter) const {
    LLVM_DEBUG(llvm::dbgs()
               << "handleUnitary: gate= " << op->getName() << '\n');

    /// If this is the first two-qubit op in the layer, route the layer
    /// and remap afterwards.

    if (!window.front().twoQubitOps.empty() &&
        op.getOperation() == window.front().twoQubitOps.front()) {
      const auto layers = to_vector(map_range(window, [](const Layer& layer) {
        return ArrayRef(layer.twoQubitIndices);
      }));
      const auto swaps = router_.route(layers, layout, *arch);
      insertSWAPs(swaps, layout, op->getLoc(), rewriter);
      history.append(swaps);
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

  AStarHeuristicRouter router_;
  std::size_t nlookahead_;
};
} // namespace mqt::ir::opt
