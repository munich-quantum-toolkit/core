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

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/Utils/WireIterator.h"

#include "mlir/IR/Types.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <utility>

namespace mlir::qco {

using Frontier = llvm::SmallMapVector<Operation*, SmallVector<size_t>, 8>;
using ReleasedOps = SmallVector<Operation*>;
using WalkProgramGraphFn =
    function_ref<WalkResult(const Frontier&, SmallVectorImpl<Operation*>&)>;

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

/// Walk the graph-like circuit IR of QCO dialect programs.
///
/// Requires verified linear QCO IR. Supported operations preserve scalar
/// qubit arity, introduce one scalar qubit, or consume qubits at wire
/// boundaries. Tensor handles do not participate in the traversal.
///
/// Depending on the template parameter, the function walks the IR in
/// topological order in forward or backward direction, respectively. Towards
/// that end, the function traverses the def-use chain of each qubit until a
/// ready operation is found. A multi-qubit gate is considered ready, if each
/// input (backward: output) qubit has been visited. The traversal considers
/// only qubit def-use dependencies. It does not order operations by classical
/// values or side effects. The signature of the callback function is:
///
/// ```c++
/// (const Frontier& frontier, SmallVectorImpl<Operation*>& released) ->
/// WalkResult
/// ```
///
/// The frontier preserves deterministic wire traversal order. The callback
/// selects frontier operations for "released" without duplicates and leaves the
/// IR and supplied iterators unchanged. The selected operations are released
/// in the next iteration. The function returns if the callback does not release
/// any operations or there are no more ready operations and thus each wire
/// points at the default sentinel. The function modifies the given wires
/// in-place.
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

      // Advance past block arguments.
      if (it != std::default_sentinel && it.operation() == nullptr) {
        std::ranges::advance(it, Traits::stride());
      }

      if (it == std::default_sentinel) {
        continue;
      }

      Operation* op = it.operation();
      if (const auto mapIt = pending.find(op); mapIt != pending.end()) {
        PendingItem& item = mapIt->second;
        item.indices_.emplace_back(i);

        if (item.ready()) {
          frontier.try_emplace(op, item.indices_);
        }
      } else {
        const auto nqubits = static_cast<size_t>(
            llvm::count_if(op->getOperandTypes(),
                           [](Type type) { return isa<QubitType>(type); }));

        // If there are fewer wires than the operation requires inputs,
        // it's impossible to release the operation. Hence, fail.

        if (nqubits > wires.size()) {
          llvm::reportFatalInternalError("more input qubits than wires");
          return;
        }

        // Single-qubit operations and sources are immediately ready.
        if (nqubits <= 1) {
          frontier.try_emplace(op, SmallVector{i});
        }

        PendingItem item(nqubits);
        item.indices_.emplace_back(i);
        pending.try_emplace(op, std::move(item));
      }
    }

    released.clear();
    const auto res = fn(frontier, released);
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
