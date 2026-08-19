/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <utility>

namespace mlir::qco {
using ReadyMap = llvm::SmallDenseMap<Operation*, SmallVector<size_t>, 8>;
using ReleasedOps = SmallVector<Operation*, 8>;
using WalkProgramGraphFn =
    function_ref<WalkResult(const ReadyMap&, ReleasedOps&)>;

/**
 * @brief Walk the graph-like circuit IR of QCO dialect programs.
 * @details
 * Depending on the template parameter, the function collects the
 * layers in forward or backward direction, respectively. Towards that end,
 * the function traverses the def-use chain of each qubit until a routing
 * boundary is found. Routing boundaries include multi-qubit gates, barriers,
 * and structured control-flow operations even when they carry only one qubit.
 * A boundary is ready once each required input wire has visited it. This
 * process is repeated until no more routing boundaries are found.
 *
 * The signature of the callback function is:
 *
 *     (const ReadyMap& ready, ReleasedOps& released) -> WalkResult
 *
 * The operations inserted into the parameter "released" determine which
 * routing boundaries are released in the next iteration.
 * If the callback returns WalkResult::skip(), all ready operations will be
 * released.
 *
 * @param wires A mutable array-ref of circuit wires (wire iterators).
 * @param fn The callback function.
 *
 * @returns success(), if all operations have been visited.
 */
template <WireDirection Direction>
LogicalResult walkProgramGraph(MutableArrayRef<WireIterator> wires,
                               WalkProgramGraphFn fn) {
  using Traits = WireTraversalTraits<Direction>;

  struct IterationStep {
    bool skip;
    size_t nqubits;
  };

  struct PendingItem {
    explicit PendingItem(const size_t nrequired) : nrequired_(nrequired) {
      indices_.reserve(nrequired);
    }

    /// Return true, if this item is ready to be released.
    [[nodiscard]] bool ready() const { return indices_.size() == nrequired_; }

    SmallVector<size_t> indices_;
    size_t nrequired_;
  };

  using PendingMap = DenseMap<Operation*, PendingItem>;

  PendingMap pending;
  pending.reserve((wires.size() + 1) / 2);

  ReadyMap ready;
  ready.reserve((wires.size() + 1) / 2);

  ReleasedOps released;

  SmallVector<size_t> curr(wires.size());
  std::iota(curr.begin(), curr.end(), 0UL);

  SmallVector<size_t> next;
  next.reserve(wires.size());

  while (!curr.empty()) {
    for (size_t i : curr) {
      auto& it = wires[i];

      if (it.operation() == nullptr) { // isa<BlockArgument>
        std::ranges::advance(it, Traits::stride());
      }

      while (Traits::isActive(it)) {
        if (const auto mapIt = pending.find(it.operation());
            mapIt != pending.end()) {
          PendingItem& item = mapIt->second;
          item.indices_.emplace_back(i);

          if (item.ready()) {
            ready.try_emplace(it.operation(), item.indices_);
          }
        } else {
          const auto [skip, nqubits] =
              TypeSwitch<Operation*, IterationStep>(it.operation())
                  .template Case<UnitaryOpInterface>(
                      [&](UnitaryOpInterface op) {
                        return IterationStep{false, op.getNumQubits()};
                      })
                  .template Case<scf::ForOp, scf::WhileOp>([&](auto op) {
                    const auto nqubits = static_cast<size_t>(
                        llvm::count_if(op.getInits(), [](Value v) {
                          return isa<QubitType>(v.getType());
                        }));
                    return IterationStep{false, nqubits};
                  })
                  .template Case<qco::IfOp>([&](qco::IfOp op) {
                    const auto nqubits = static_cast<size_t>(
                        llvm::count_if(op.getQubits(), [](Value v) {
                          return isa<QubitType>(v.getType());
                        }));
                    return IterationStep{false, nqubits};
                  })
                  .template Case<qco::IndexSwitchOp>(
                      [&](qco::IndexSwitchOp op) {
                        const auto nqubits = static_cast<size_t>(
                            llvm::count_if(op.getTargets(), [](Value v) {
                              return isa<QubitType>(v.getType());
                            }));
                        return IterationStep{false, nqubits};
                      })
                  .template Case<ResetOp, MeasureOp>(
                      [&](auto) { return IterationStep{false, 1}; })
                  .template Case<AllocOp, StaticOp, SinkOp, YieldOp,
                                 qtensor::ExtractOp, qtensor::InsertOp,
                                 scf::YieldOp, scf::ConditionOp>(
                      [&](auto) { return IterationStep{true, 0}; })
                  .Default([&](Operation* op) {
                    const auto name = op->getName().getStringRef();
                    reportFatalInternalError("unknown op: " + name);
                    return IterationStep{false, 0};
                  });

          const bool isControlFlowBoundary =
              isa<scf::ForOp, scf::WhileOp, qco::IfOp, qco::IndexSwitchOp>(
                  it.operation());
          if (skip || (nqubits == 1 && !isControlFlowBoundary)) {
            std::ranges::advance(it, Traits::stride());
            continue;
          }

          // If there are fewer wires than the operation requires inputs,
          // it's impossible to release the operation. Hence, fail.

          if (nqubits > wires.size()) {
            return failure();
          }

          // Insert the multi-qubit op to the pending map.
          // The caller decides if this op should be released.
          PendingItem item(nqubits);
          item.indices_.emplace_back(i);
          auto [pendingIt, inserted] =
              pending.try_emplace(it.operation(), std::move(item));
          assert(inserted);
          if (pendingIt->second.ready()) {
            ready.try_emplace(it.operation(), pendingIt->second.indices_);
          }
        }

        break; // Stop at multi-qubit unitary.
      }
    }

    released.clear();
    const auto res = std::invoke(fn, ready, released);
    if (res.wasInterrupted()) {
      return failure();
    }

    if (res.wasSkipped()) {
      released.clear();
      for (Operation* op : ready.keys()) {
        released.emplace_back(op);
      }
    }

    for (Operation* op : released) {
      const auto mapIt = pending.find(op);
      assert(mapIt != pending.end());

      for (size_t i : mapIt->second.indices_) {
        std::ranges::advance(wires[i], Traits::stride());
        next.emplace_back(i);
      }

      pending.erase(mapIt);
      ready.erase(op);
    }

    curr.swap(next);
    next.clear();
  }

  return success();
}
} // namespace mlir::qco
