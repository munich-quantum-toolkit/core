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

#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Utils/WireIterator.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <utility>

namespace mlir::qco {

using Frontier = llvm::SmallMapVector<Operation*, SmallVector<size_t>, 8>;
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

/// Reusable traversal storage. A scratch instance must not be shared by walks
/// that overlap; completed walks retain capacity, but no IR handles.
struct WalkProgramGraphScratch {
  DenseMap<Operation*, impl::PendingItem> pending;
  Frontier frontier;
  ReleasedOps released;
  SmallVector<size_t> curr;
  SmallVector<size_t> next;

  void clear() {
    pending.clear();
    frontier.clear();
    released.clear();
    curr.clear();
    next.clear();
  }
};

/// Walk the graph-like circuit IR of QCO dialect programs.
/// Depending on the template parameter, the function walks the IR in
/// topological order in forward or backward direction, respectively. Towards
/// that end, the function traverses the def-use chain of each qubit until a
/// ready operation is found. A multi-qubit gate is considered ready, if each
/// input (backward: output) qubit has been visited.
/// The traversal considers only qubit def-use dependencies. It does not order
/// operations by classical values or side effects.
/// The signature of the callback function is:
///
///     (const Frontier& frontier, ReleasedOps& released) -> WalkResult
///
/// The frontier preserves deterministic wire traversal order.
/// The operations inserted into the "released" vector determine which
/// operations are released in the next iteration. The function returns if the
/// callback does not release any operations or there are no more ready
/// operations and thus each wire points at the default sentinel.
/// The function modifies the given wires in-place.
template <WireDirection Direction>
void walkProgramGraph(MutableArrayRef<WireIterator> wires,
                      WalkProgramGraphFn fn, WalkProgramGraphScratch& scratch) {
  using namespace impl;
  using Traits = WireTraversalTraits<Direction>;

  scratch.clear();
  const auto cleanup = llvm::make_scope_exit([&] { scratch.clear(); });
  auto& [pending, frontier, released, curr, next] = scratch;
  pending.reserve((wires.size() + 1) / 2);
  frontier.reserve((wires.size() + 1) / 2);
  curr.resize(wires.size());
  std::iota(curr.begin(), curr.end(), 0UL);
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
            frontier.try_emplace(it.operation(), std::move(item.indices_));
            pending.erase(mapIt);
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
          } else {
            PendingItem item(nqubits);
            item.indices_.emplace_back(i);
            pending.try_emplace(it.operation(), std::move(item));
          }
        }

        break;
      }
    }

    released.clear();
    const auto res = std::invoke(fn, frontier, released);
    if (res.wasInterrupted() || res.wasSkipped()) {
      return;
    }

    const bool releaseAll = released.size() == frontier.size();
    for (Operation* op : released) {
      const auto mapIt = frontier.find(op);
      assert(mapIt != frontier.end());

      for (size_t i : mapIt->second) {
        std::ranges::advance(wires[i], Traits::stride());
        next.emplace_back(i);
      }

      if (!releaseAll) {
        frontier.erase(mapIt);
      }
    }
    if (releaseAll) {
      frontier.clear();
    }

    curr.swap(next);
    next.clear();
  }
}

template <WireDirection Direction>
void walkProgramGraph(MutableArrayRef<WireIterator> wires,
                      WalkProgramGraphFn fn) {
  WalkProgramGraphScratch scratch;
  walkProgramGraph<Direction>(wires, fn, scratch);
}
} // namespace mlir::qco
