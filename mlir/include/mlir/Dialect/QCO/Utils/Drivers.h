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
namespace details {
struct PendingItem {
  explicit PendingItem(const size_t nrequired) : nrequired(nrequired) {
    indices.reserve(nrequired);
  }

  /// Return true, if this item is ready to be released.
  [[nodiscard]] bool ready() const { return indices.size() == nrequired; }

  SmallVector<size_t> indices;
  size_t nrequired;
};

using PendingMap = DenseMap<Operation*, PendingItem>;
} // namespace details

using ReadyVec = SmallVector<std::pair<Operation*, SmallVector<size_t>>, 0>;
using ReleasedOps = SmallVector<Operation*, 8>;
using WalkProgramGraphFn =
    function_ref<WalkResult(const ReadyVec&, ReleasedOps&)>;

/**
 * @brief Walk the graph-like circuit IR of QCO dialect programs.
 * @details
 * Depending on the template parameter, the function collects the
 * layers in forward or backward direction, respectively. Towards that end,
 * the function traverses the def-use chain of each qubit until a multi-qubit
 * gate (including barriers) is found. If each input qubit of a multi-qubit gate
 * is visited, it is considered ready. This process is repeated until no more
 * multi-qubit gates are found anymore.
 *
 * The signature of the callback function is:
 *
 *     (const ReadyRange& ready, ReleasedOps& released) -> WalkResult
 *
 * The operations inserted into the parameter "released" determine which
 * multi-qubit gates are released in next iteration.
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

  using IterationStep = std::pair</*skip= */ bool, /*nqubits= */ size_t>;

  ReleasedOps released;
  details::PendingMap pending;
  pending.reserve((wires.size() + 1) / 2);

  ReadyVec ready;
  ready.reserve((wires.size() + 1) / 2);

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
          details::PendingItem& item = mapIt->second;
          item.indices.emplace_back(i);
          if (item.ready()) {
            ready.emplace_back(it.operation(), item.indices);
          }
        } else {
          const auto [skip, nqubits] =
              TypeSwitch<Operation*, IterationStep>(it.operation())
                  .template Case<UnitaryOpInterface>(
                      [&](UnitaryOpInterface op) {
                        return std::make_pair(false, op.getNumQubits());
                      })
                  .template Case<scf::ForOp, scf::WhileOp>([&](auto op) {
                    const auto nqubits =
                        llvm::count_if(op.getInits(), [](Value v) {
                          return isa<QubitType>(v.getType());
                        });
                    return std::make_pair(false, nqubits);
                  })
                  .template Case<qco::IfOp>([&](qco::IfOp op) {
                    const auto nqubits =
                        llvm::count_if(op.getQubits(), [](Value v) {
                          return isa<QubitType>(v.getType());
                        });
                    return std::make_pair(false, nqubits);
                  })
                  .template Case<qco::IndexSwitchOp>(
                      [&](qco::IndexSwitchOp op) {
                        const auto nqubits =
                            llvm::count_if(op.getTargets(), [](Value v) {
                              return isa<QubitType>(v.getType());
                            });
                        return std::make_pair(false, nqubits);
                      })
                  .template Case<ResetOp, MeasureOp>(
                      [&](auto) { return std::make_pair(false, 1); })
                  .template Case<AllocOp, StaticOp, SinkOp, YieldOp,
                                 qtensor::ExtractOp, qtensor::InsertOp,
                                 scf::YieldOp, scf::ConditionOp>(
                      [&](auto) { return std::make_pair(true, 0); })
                  .Default([&](Operation* op) {
                    const auto name = op->getName().getStringRef();
                    reportFatalInternalError("unknown op: " + name);
                    return std::make_pair(false, 0);
                  });

          if (skip || nqubits == 1) {
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
          details::PendingItem item(nqubits);
          item.indices.emplace_back(i);
          pending.try_emplace(it.operation(), std::move(item));
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
      for (const auto& [op, _] : ready) {
        released.emplace_back(op);
      }
    }

    for (Operation* op : released) {
      const auto mapIt = pending.find(op);
      assert(mapIt != pending.end());

      for (size_t i : mapIt->second.indices) {
        std::ranges::advance(wires[i], Traits::stride());
        next.emplace_back(i);
      }

      pending.erase(mapIt);
    }

    curr.swap(next);
    next.clear();
    ready.clear();
  }

  return success();
}
} // namespace mlir::qco
