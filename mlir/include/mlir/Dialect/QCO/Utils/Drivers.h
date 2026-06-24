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
#include "mlir/Dialect/QCO/Utils/Qubits.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

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
            const auto prevQ = cast<TypedValue<QubitType>>(prevV);
            const auto nextQ = cast<TypedValue<QubitType>>(nextV);
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

using ReleasedOps = SmallVector<Operation*, 8>;
using PendingWiresMap = DenseMap<Operation*, SmallVector<size_t>>;

namespace impl {
/// Return the number of qubits a operation produces/consumes.
inline size_t getNumQubits(Operation* op) {
  return TypeSwitch<Operation*, size_t>(op)
      .Case<UnitaryOpInterface>(
          [&](UnitaryOpInterface op) { return op.getNumQubits(); })
      .Case<scf::ForOp>([&](scf::ForOp op) { return op.getInits().size(); })
      .Case<scf::WhileOp>([&](scf::WhileOp op) { return op.getInits().size(); })
      .Case<qco::IfOp>([&](qco::IfOp op) { return op.getQubits().size(); })
      .Case<AllocOp, StaticOp, ResetOp, MeasureOp, SinkOp, YieldOp,
            qtensor::ExtractOp, qtensor::InsertOp, scf::YieldOp>(
          [&](auto) { return 1; })
      .Default([&](Operation* op) {
        const auto name = op->getName().getStringRef();
        reportFatalInternalError("unknown op: " + name);
        return 0;
      });
}
} // namespace impl

struct IsReady {
  bool operator()(PendingWiresMap::value_type& kv) const {
    const auto npending = kv.second.size();
    return impl::getNumQubits(kv.first) == npending;
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

      if (it.operation() == nullptr) { // isa<BlockArgument>
        std::ranges::advance(it, Traits::stride());
      }

      while (Traits::isActive(it)) {
        assert(it.operation() != nullptr);
        const auto nqubits = impl::getNumQubits(it.operation());

        assert(nqubits != 0);
        if (nqubits == 1) {
          std::ranges::advance(it, Traits::stride());
          continue;
        }

        assert(nqubits >= 2);

        // If there are fewer wires than the operation requires inputs,
        // it's impossible to release the operation. Hence, fail.

        if (nqubits > wires.size()) {
          return failure();
        }

        // Insert the unitary to the pending map.
        // The caller decides if this op should be released.

        const auto [mapIt, inserted] = pending.try_emplace(it.operation());
        auto& indices = mapIt->second;

        if (inserted) {
          indices.reserve(nqubits);
        }

        indices.emplace_back(i);

        break; // Stop at multi-qubit unitary.
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
