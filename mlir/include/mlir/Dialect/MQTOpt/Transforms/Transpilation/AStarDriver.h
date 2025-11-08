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

#include "llvm/ADT/iterator_range.h"

#include <llvm/ADT/STLExtras.h>
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
  /// Operations that can be executed before the two-qubit gates.
  SmallVector<Operation*> ops;
  /// The two-qubit gates.
  SmallVector<Operation*> twoQubitOps;
  /// The program indices of the two-qubit gates within this layer.
  SmallVector<QubitIndexPair> twoQubitIndices;
  /// Operations that will be executable whenever all gates in the layer are
  /// executable.
  SmallVector<Operation*> blockOps;

  /// @returns true if the layer has no gates to route.
  [[nodiscard]] bool empty() const { return twoQubitIndices.empty(); }

  /// @returns the scheduled operations in (to be) reordered order.
  [[nodiscard]] auto getOps() {
    return concat<Operation*>(ops, twoQubitOps, blockOps);
  }
};

using Schedule = SmallVector<Layer, 0>;

class Scheduler {
public:
  explicit Scheduler(Region* region) : region(region) {}

  /**
   * @brief Starting from the given layout, schedule all operations and divide
   * the circuit into parallelly executable layers.
   * @returns the schedule.
   */
  [[nodiscard]] Schedule schedule(Layout layout) {
    /// Worklist of qubits.
    SmallVector<Value> qubits(layout.getHardwareQubits());
    /// Set of two-qubit gates seen at least once.
    DenseSet<Operation*> openTwoQubit;

    Schedule schedule;
    while (true) {
      const auto result = getNextLayer(qubits, openTwoQubit, layout);
      if (result.layer.empty()) {
        break;
      }

      schedule.emplace_back(result.layer);
      qubits = result.qubits;
    }

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
    /// The input value of the unitary interface.
    Value q = nullptr;
    /// The unitary interface.
    UnitaryInterface u = nullptr;
    /// All operations visited until the two-qubit gate.
    SmallVector<Operation*> ops;
    /// @returns true iff. the advancement hit a two-qubit gate.
    [[nodiscard]] bool valid() const { return u != nullptr; }
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
                                             DenseSet<Operation*>& openTwoQubit,
                                             Layout& layout) {
    NextLayerResult bundle;
    bundle.qubits.reserve(qubits.size());

    for (const Value q : qubits) {
      if (q.use_empty()) {
        continue;
      }

      const auto result = advanceToTwoQubitGate(q, layout);
      if (!result.valid()) {
        continue;
      }

      if (q != result.q) {
        layout.remapQubitValue(q, result.q);
      }

      bundle.layer.ops.append(result.ops);

      if (!openTwoQubit.insert(result.u).second) {
        const auto ins = getIns(result.u);
        const auto outs = getOuts(result.u);

        bundle.layer.twoQubitOps.emplace_back(result.u);
        bundle.layer.twoQubitIndices.emplace_back(
            layout.lookupProgramIndex(ins.first),
            layout.lookupProgramIndex(ins.second));

        layout.remapQubitValue(ins.first, outs.first);
        layout.remapQubitValue(ins.second, outs.second);

        const auto blockResult = skipTwoQubitBlock(outs, layout);
        bundle.layer.blockOps.append(blockResult.ops);

        if (blockResult.outs.first != nullptr &&
            !blockResult.outs.first.use_empty()) {
          bundle.qubits.push_back(blockResult.outs.first);
        }

        if (blockResult.outs.second != nullptr &&
            !blockResult.outs.second.use_empty()) {
          bundle.qubits.push_back(blockResult.outs.second);
        }

        openTwoQubit.erase(result.u);
      }
    }

    return bundle;
  }

  /**
   * @returns todo
   */
  AdvanceResult advanceToTwoQubitGate(const Value q, const Layout& layout) {
    AdvanceResult result;

    Value head = q;
    while (true) {
      if (head.use_empty()) { // No two-qubit gate found.
        break;
      }

      Operation* user = getUserInRegion(head, region);
      if (user == nullptr) { // No two-qubit gate found.
        break;
      }

      result.ops.push_back(user);

      bool endOfRegion = false;
      TypeSwitch<Operation*>(user)
          /// MQT
          /// BarrierOp is a UnitaryInterface, however, requires special care.
          .Case<BarrierOp>([&](BarrierOp op) {
            for (const auto [in, out] :
                 llvm::zip_equal(op.getInQubits(), op.getOutQubits())) {
              if (in == head) {
                head = out;
                return;
              }
            }
            llvm_unreachable("head must be in barrier");
          })
          .Case<UnitaryInterface>([&](UnitaryInterface op) {
            if (isTwoQubitGate(op)) {
              result.u = op;
              return; // Found a two-qubit gate, stop advancing head.
            }
            // Otherwise, advance head.
            head = op.getOutQubits().front();
          })
          .Case<ResetOp>([&](ResetOp op) { head = op.getOutQubit(); })
          .Case<MeasureOp>([&](MeasureOp op) { head = op.getOutQubit(); })

          /// SCF
          /// The scf funcs assume that the first n results are the hw qubits.
          .Case<scf::ForOp>([&](scf::ForOp op) {
            head = op->getResult(layout.lookupHardwareIndex(q));
          })
          .Case<scf::IfOp>([&](scf::IfOp op) {
            head = op->getResult(layout.lookupHardwareIndex(q));
          })
          .Case<scf::YieldOp>([&](scf::YieldOp) { endOfRegion = true; })
          .Default([&]([[maybe_unused]] Operation* op) {
            LLVM_DEBUG({
              llvm::dbgs() << "unknown operation in def-use chain: ";
              op->dump();
            });
            llvm_unreachable("unknown operation in def-use chain");
          });

      if (result.u) {          // Two-qubit gate found.
        result.ops.pop_back(); // Remove two-qubit gate from op list.
        result.q = head;
        break;
      }

      if (endOfRegion) {
        break;
      }
    }

    return result;
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
        const auto result = advanceToTwoQubitGate(q, layout);
        if (!result.valid()) {
          heads[i] = nullptr;
          stop = true;
          break;
        }

        if (q != result.q) {
          layout.remapQubitValue(q, result.q);
        }

        blockResult.ops.append(result.ops);

        heads[i] = result.q;
        gates[i] = result.u;
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

  Region* region;
};

class AStarDriver final : public RoutingDriverBase {
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

    return rewrite(func.getBody(), layout, rewriter);
  }

  LogicalResult rewrite(Region& region, Layout& layout,
                        PatternRewriter& rewriter) const {
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

        if (prev != nullptr) {
          if (isa<scf::ForOp>(prev) || isa<scf::IfOp>(prev) ||
              isa<scf::ForOp>(curr) || isa<scf::IfOp>(curr)) {
            continue;
          }

          rewriter.moveOpAfter(curr, prev);
        }

        const OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(curr);

        const auto res =
            TypeSwitch<Operation*, WalkResult>(curr)
                /// mqtopt Dialect
                .Case<UnitaryInterface>([&](UnitaryInterface op) {
                  return handleUnitary(op, layout, window, rewriter);
                })
                .Case<ResetOp>(
                    [&](ResetOp op) { return handleReset(op, layout); })
                .Case<MeasureOp>(
                    [&](MeasureOp op) { return handleMeasure(op, layout); })
                /// built-in Dialect
                .Case<ModuleOp>([&]([[maybe_unused]] ModuleOp op) {
                  return WalkResult::advance();
                })
                /// func Dialect
                .Case<func::ReturnOp>([&]([[maybe_unused]] func::ReturnOp op) {
                  return WalkResult::advance();
                })
                /// scf Dialect
                // .Case<scf::ForOp>([&](scf::ForOp op) {
                //   return handleFor(op, layout, rewriter);
                // })
                // .Case<scf::IfOp>([&](scf::IfOp op) {
                //   return handleIf(op, layout, rewriter);
                // })
                // .Case<scf::YieldOp>([&](scf::YieldOp op) {
                //   return handleYield(op, layout, history, rewriter);
                // })
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
   * @brief Ensures the executability of two-qubit gates on the given target
   * architecture by inserting SWAPs.
   */
  WalkResult handleUnitary(UnitaryInterface op, Layout& layout,
                           const ArrayRef<const Layer> window,
                           PatternRewriter& rewriter) const {
    LLVM_DEBUG(llvm::dbgs()
               << "handleUnitary: gate= " << op->getName() << '\n');
    /// If this is the first two-qubit op in the layer, route the layer
    /// and remap afterwards.
    if (op.getOperation() == window.front().twoQubitOps.front()) {
      const auto layers = to_vector(map_range(window, [](const Layer& layer) {
        return ArrayRef(layer.twoQubitIndices);
      }));
      const auto swaps = router_.route(layers, layout, *arch);
      insertSWAPs(swaps, layout, op->getLoc(), rewriter);
    }

    for (const auto [in, out] :
         llvm::zip(op.getAllInQubits(), op.getAllOutQubits())) {
      layout.remapQubitValue(in, out);
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
