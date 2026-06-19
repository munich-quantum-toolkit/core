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

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <utility>

namespace mlir::qco {

using ReleasedOps = SmallVector<Operation*, 8>;
using PendingWiresMap = DenseMap<Operation*, SmallVector<size_t, 2>>;

struct IsReady {
  bool operator()(PendingWiresMap::value_type& kv) const {
    const auto npending = kv.second.size();
    return TypeSwitch<Operation*, bool>(kv.first)
        .Case<UnitaryOpInterface>(
            [&](auto& op) { return op.getNumQubits() == npending; })
        .template Case<scf::ForOp>(
            [&](auto& op) { return op.getInits().size() == npending; })
        .Default([&](Operation* op) {
          const auto name = op->getName().getStringRef();
          reportFatalInternalError("unknown pending op: " + name);
          return false;
        });
  }
};

using ReadyRange =
    decltype(make_filter_range(std::declval<PendingWiresMap&>(), IsReady{}));

using WalkProgramGraphFn =
    function_ref<WalkResult(const ReadyRange&, ReleasedOps&)>;

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

  ReleasedOps released;

  PendingWiresMap pending;
  pending.reserve(wires.size());

  SmallVector<size_t> curr(wires.size());
  std::iota(curr.begin(), curr.end(), 0UL);

  SmallVector<size_t> next;
  next.reserve(wires.size());

  while (!curr.empty()) {
    for (size_t i : curr) {
      auto& it = wires[i];
      while (Traits::isActive(it)) {

        if (it.qubit() != nullptr && isa<BlockArgument>(it.qubit())) {
          std::ranges::advance(it, Traits::stride());
          continue;
        }

        assert(it.operation() != nullptr);

        const auto res =
            TypeSwitch<Operation*, WalkResult>(it.operation())
                .template Case<UnitaryOpInterface>([&](auto& op) {
                  // If there are fewer wires than the unitary requires inputs,
                  // it's impossible to release the operation. Hence, fail.
                  if (op.getNumQubits() > wires.size()) {
                    return WalkResult::interrupt();
                  }

                  if (op.getNumQubits() == 1) {
                    std::ranges::advance(it, Traits::stride());
                    return WalkResult::advance();
                  }

                  // Insert the unitary to the pending map.
                  // The caller decides if this op should be released.
                  const auto [mapIt, inserted] = pending.try_emplace(op);
                  auto& indices = mapIt->second;

                  if (inserted) {
                    indices.reserve(op.getNumQubits());
                  }

                  indices.emplace_back(i);

                  return WalkResult::skip(); // Stop at multi-qubit unitary.
                })
                .template Case<scf::ForOp>([&](scf::ForOp& op) {
                  // If there are fewer wires than the loop requires inputs,
                  // it's impossible to release the operation. Hence, fail.
                  if (op.getInits().size() > wires.size()) {
                    return WalkResult::interrupt();
                  }

                  if (op.getInits().size() == 1) {
                    std::ranges::advance(it, Traits::stride());
                    return WalkResult::advance();
                  }

                  // Insert the loop to the pending map.
                  // The caller decides if this op should be released.
                  const auto [mapIt, inserted] = pending.try_emplace(op);
                  auto& indices = mapIt->second;

                  if (inserted) {
                    indices.reserve(op.getInits().size());
                  }

                  indices.emplace_back(i);

                  return WalkResult::skip(); // Stop at multi-qubit loop.
                })
                // AllocOp, StaticOp, and qtensor::ExtractOp are only reachable
                // on the forward path; backward isActive() halts before
                // reaching them (decrementing at a source op is a no-op).
                .template Case<AllocOp, StaticOp, ResetOp, MeasureOp, SinkOp,
                               YieldOp, scf::YieldOp, qtensor::ExtractOp,
                               qtensor::InsertOp>([&](auto) {
                  std::ranges::advance(it, Traits::stride());
                  return WalkResult::advance();
                })
                .Default([&](Operation* op) -> WalkResult {
                  const auto name = op->getName().getStringRef();
                  reportFatalInternalError("unknown op encountered: " + name);
                });

        if (res.wasSkipped()) {
          break;
        }

        if (res.wasInterrupted()) {
          return failure();
        }
      }
    }

    released.clear();
    const auto ready = make_filter_range(pending, IsReady{});
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

      for (size_t i : mapIt->second) {
        std::ranges::advance(wires[i], Traits::stride());
        next.emplace_back(i);
      }

      pending.erase(mapIt);
    }

    curr.swap(next);
    next.clear();
  }

  return success();
}
} // namespace mlir::qco
