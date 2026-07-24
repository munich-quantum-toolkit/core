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

using ReleasedOps = SmallVector<Operation*, 8>;
using PendingWiresMap = DenseMap<Operation*, SmallVector<size_t>>;

namespace impl {

/// Return the number of qubit arguments of unitary-like operation.
inline size_t getNumQubitArgs(Operation* op) {
  return TypeSwitch<Operation*, size_t>(op)
      .Case<UnitaryOpInterface>(
          [&](UnitaryOpInterface op) { return op.getNumQubits(); })
      .Case<scf::ForOp, scf::WhileOp>([&](auto op) {
        return llvm::count_if(
            op.getInits(), [](Value v) { return isa<QubitType>(v.getType()); });
      })
      .Case<qco::IfOp>([&](qco::IfOp op) {
        return llvm::count_if(op.getQubits(), [](Value v) {
          return isa<QubitType>(v.getType());
        });
      })
      .Case<qco::IndexSwitchOp>([&](qco::IndexSwitchOp op) {
        return llvm::count_if(op.getTargets(), [](Value v) {
          return isa<QubitType>(v.getType());
        });
      })
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
    return impl::getNumQubitArgs(kv.first) == npending;
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

        // For source-like (AllocOp, StaticOp, qtensor::ExtractOp),
        // sink-like (SinkOp, YieldOp, qtensor::InsertOp, scf::YieldOp,
        // scf::ConditionOp), and one-qubit non-unitary (ResetOp, MeasureOp)
        // operations, simply advance the iterator.

        if (isa<AllocOp, StaticOp, ResetOp, MeasureOp, SinkOp, YieldOp,
                qtensor::ExtractOp, qtensor::InsertOp, scf::YieldOp,
                scf::ConditionOp>(it.operation())) {
          std::ranges::advance(it, Traits::stride());
          continue;
        }

        const auto nqubits = impl::getNumQubitArgs(it.operation());

        // Advance past one-qubit operations.

        if (nqubits == 1) {
          std::ranges::advance(it, Traits::stride());
          continue;
        }

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
