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
#include <limits>
#include <numeric>
#include <utility>

namespace mlir::qco {

using Frontier = llvm::SmallDenseMap<Operation*, SmallVector<size_t>, 8>;
using ReleasedOps = SmallVector<Operation*, 8>;
using WalkProgramGraphFn =
    function_ref<WalkResult(const Frontier&, ReleasedOps&)>;

namespace impl {
struct PendingItem {
  explicit PendingItem(const size_t nrequired) : nrequired_(nrequired) {
    indices_.reserve(nrequired);
  }

  /// Return true, if this item is ready to be released.
  [[nodiscard]] bool ready() const { return indices_.size() == nrequired_; }

  SmallVector<size_t> indices_;
  size_t nrequired_;
};
} // namespace impl

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
 *     (const Frontier& frontier, ReleasedOps& released) -> WalkResult
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
void walkProgramGraph(MutableArrayRef<WireIterator> wires,
                      WalkProgramGraphFn fn) {
  using namespace impl;
  using Traits = WireTraversalTraits<Direction>;

  DenseMap<Operation*, PendingItem> pending;
  pending.reserve((wires.size() + 1) / 2);

  Frontier frontier;
  frontier.reserve((wires.size() + 1) / 2);

  ReleasedOps released;

  SmallVector<size_t> curr(wires.size());
  std::iota(curr.begin(), curr.end(), 0UL);

  SmallVector<size_t> next;
  next.reserve(wires.size());

  while (!curr.empty()) {
    for (const auto i : curr) {
      auto& it = wires[i];

      while (it != std::default_sentinel) {
        if (it.operation() == nullptr) { // isa<BlockArgument>
          std::ranges::advance(it, Traits::stride());
          continue;
        }

        if (const auto mapIt = pending.find(it.operation());
            mapIt != pending.end()) {
          PendingItem& item = mapIt->second;
          item.indices_.emplace_back(i);

          if (item.ready()) {
            frontier.try_emplace(it.operation(), item.indices_);
          }
        } else {
          const auto nqubits =
              TypeSwitch<Operation*, size_t>(it.operation())
                  .template Case<UnitaryOpInterface>(
                      [&](UnitaryOpInterface op) { return op.getNumQubits(); })
                  .template Case<scf::ForOp, scf::WhileOp>([&](auto op) {
                    const auto nqubits = static_cast<size_t>(
                        llvm::count_if(op.getInits(), [](Value v) {
                          return isa<QubitType>(v.getType());
                        }));
                    return nqubits;
                  })
                  .template Case<qco::IfOp>([&](qco::IfOp op) {
                    return static_cast<size_t>(
                        llvm::count_if(op.getQubits(), [](Value v) {
                          return isa<QubitType>(v.getType());
                        }));
                  })
                  .template Case<qco::IndexSwitchOp>([](qco::IndexSwitchOp op) {
                    return static_cast<size_t>(
                        llvm::count_if(op.getTargets(), [](Value v) {
                          return isa<QubitType>(v.getType());
                        }));
                  })
                  .template Case<AllocOp, StaticOp, SinkOp, qtensor::ExtractOp,
                                 qtensor::InsertOp, ResetOp, MeasureOp>(
                      [](auto) { return 1; })
                  .template Case<YieldOp>([](YieldOp op) {
                    return static_cast<size_t>(
                        llvm::count_if(op.getTargets(), [](Value v) {
                          return isa<QubitType>(v.getType());
                        }));
                  })
                  .template Case<scf::YieldOp>([](scf::YieldOp op) {
                    return static_cast<size_t>(
                        llvm::count_if(op.getResults(), [](Value v) {
                          return isa<QubitType>(v.getType());
                        }));
                  })
                  .template Case<scf::ConditionOp>([](scf::ConditionOp op) {
                    return static_cast<size_t>(
                        llvm::count_if(op.getArgs(), [](Value v) {
                          return isa<QubitType>(v.getType());
                        }));
                  })
                  .Default([&](Operation* op) {
                    const auto name = op->getName().getStringRef();
                    reportFatalInternalError("unknown op: " + name);
                    return std::numeric_limits<size_t>::max();
                  });

          // If there are fewer wires than the operation requires inputs,
          // it's impossible to release the operation. Hence, fail.

          if (nqubits > wires.size()) {
            llvm::reportFatalInternalError("more input qubits than wires");
            return;
          }

          // One-qubit gates are immediately ready.
          // Hence, add them to the frontier.

          if (nqubits == 1) {
            frontier.try_emplace(it.operation(), SmallVector{i});
          }

          PendingItem item(nqubits);
          item.indices_.emplace_back(i);
          pending.try_emplace(it.operation(), std::move(item));
        }

        break;
      }
    }

    released.clear();
    const auto res = std::invoke(fn, frontier, released);
    if (res.wasInterrupted() || res.wasSkipped()) {
      return;
    }

    for (Operation* op : released) {
      const auto mapIt = pending.find(op);
      assert(mapIt != pending.end());

      for (size_t i : mapIt->second.indices_) {
        std::ranges::advance(wires[i], Traits::stride());
        next.emplace_back(i);
      }

      pending.erase(mapIt);
      frontier.erase(op);
    }

    curr.swap(next);
    next.clear();
  }
}
} // namespace mlir::qco
