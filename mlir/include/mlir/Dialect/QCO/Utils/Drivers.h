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

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Qubits.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <utility>

namespace mlir::qco {

using WalkProgramFn = function_ref<WalkResult(Operation*, Qubits&)>;

/**
 * @brief Perform top-down non-recursive walk of all operations within a
 * region of a quantum program and apply a callback function.
 * @details The signature of the callback function is:
 *
 *     (Operation*, Qubits& q) -> WalkResult
 *
 * where the Qubits object tracks the front of qubit SSA values.
 * Depending on the template parameter, the callback is executed before or after
 * updating the Qubits state.
 * @param region The targeted region.
 * @param fn The callback function.
 * @returns success(), if all operations have been visited.
 */
template <WalkOrder Order = WalkOrder::PreOrder>
LogicalResult walkProgram(Region& region, const WalkProgramFn& fn) {
  Qubits qubits;
  for (Operation& curr : region.getOps()) {
    if constexpr (Order == WalkOrder::PreOrder) {
      if (fn(&curr, qubits).wasInterrupted()) {
        return failure();
      }
    }

    TypeSwitch<Operation*>(&curr)
        .template Case<StaticOp>(
            [&](StaticOp op) { qubits.add(op.getQubit(), op.getIndex()); })
        .template Case<AllocOp>([&](AllocOp op) { qubits.add(op.getResult()); })
        .template Case<UnitaryOpInterface>([&](UnitaryOpInterface& op) {
          for (const auto& [prevV, nextV] :
               llvm::zip(op.getInputQubits(), op.getOutputQubits())) {
            const auto prevQ = llvm::cast<TypedValue<QubitType>>(prevV);
            const auto nextQ = llvm::cast<TypedValue<QubitType>>(nextV);
            qubits.remap(prevQ, nextQ);
          }
        })
        .template Case<ResetOp>([&](ResetOp op) {
          qubits.remap(op.getQubitIn(), op.getQubitOut());
        })
        .template Case<MeasureOp>([&](MeasureOp op) {
          qubits.remap(op.getQubitIn(), op.getQubitOut());
        })
        .template Case<SinkOp>(
            [&](SinkOp op) { qubits.remove(op.getQubit()); });

    if constexpr (Order == WalkOrder::PostOrder) {
      if (fn(&curr, qubits).wasInterrupted()) {
        return failure();
      }
    }
  }

  return success();
}

using ReleasedOps = SmallVector<UnitaryOpInterface, 8>;
using PendingWiresMap =
    DenseMap<UnitaryOpInterface, SmallVector<std::size_t, 2>>;

struct IsReady {
  bool operator()(PendingWiresMap::value_type& kv) const {
    return kv.second.size() == kv.first.getNumQubits();
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

  SmallVector<std::size_t> curr(wires.size());
  std::iota(curr.begin(), curr.end(), 0UL);

  SmallVector<std::size_t> next;
  next.reserve(wires.size());

  while (!curr.empty()) {
    for (std::size_t i : curr) {
      auto& it = wires[i];
      while (Traits::isActive(it)) {
        const auto res =
            TypeSwitch<Operation*, WalkResult>(it.operation())
                .template Case<UnitaryOpInterface>([&](UnitaryOpInterface& op) {
                  // If there are fewer wires than the qubit requires inputs,
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

                  return WalkResult::skip(); // Stop at multi-qubit gate.
                })
                // AllocOp, StaticOp, and qtensor::ExtractOp are only reachable
                // on the forward path; backward isActive() halts before
                // reaching them (decrementing at a source op is a no-op).
                .template Case<AllocOp, StaticOp, qtensor::ExtractOp, ResetOp,
                               MeasureOp, SinkOp, qtensor::InsertOp>([&](auto) {
                  std::ranges::advance(it, Traits::stride());
                  return WalkResult::advance();
                })
                .Default([&](Operation* op) -> WalkResult {
                  const auto name = op->getName().getStringRef();
                  report_fatal_error("unknown op encountered: " + name);
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

    for (const UnitaryOpInterface& op : released) {
      const auto mapIt = pending.find(op);
      assert(mapIt != pending.end());

      for (std::size_t i : mapIt->second) {
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
