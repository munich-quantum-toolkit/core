/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Drivers.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <utility>

namespace mlir::qco {

using namespace mlir::qtensor;

enum class CircuitWalkResult : std::uint8_t { Advance, Hold, Fail };

bool proceedOnWire(const WireIterator& it, WalkDirection direction) {
  if (direction == WalkDirection::Forward) {
    return it != std::default_sentinel;
  }

  if (it.operation() == nullptr) {
    return false;
  }

  return !isa<qco::AllocOp>(it.operation()) && !isa<StaticOp>(it.operation()) &&
         !isa<qtensor::ExtractOp>(it.operation());
}

void Qubits::add(TypedValue<QubitType> q) { add(q, indexToValue_.size()); }

void Qubits::add(TypedValue<QubitType> q, std::size_t index) {
  indexToValue_.try_emplace(index, q);
  valueToIndex_.try_emplace(q, index);
}

void Qubits::remap(TypedValue<QubitType> prev, TypedValue<QubitType> next,
                   const WalkDirection& direction) {
  if (direction == WalkDirection::Backward) {
    std::swap(prev, next);
  }

  assert(valueToIndex_.contains(prev));
  const auto index = valueToIndex_.lookup(prev);

  valueToIndex_.erase(prev);
  valueToIndex_.try_emplace(next, index);
  indexToValue_[index] = next;
}

void Qubits::remap(UnitaryOpInterface op, const WalkDirection& direction) {
  for (const auto& [in, out] :
       llvm::zip_equal(op.getInputQubits(), op.getOutputQubits())) {
    remap(cast<TypedValue<QubitType>>(in), cast<TypedValue<QubitType>>(out),
          direction);
  }
}

void Qubits::remove(TypedValue<QubitType> q) {
  assert(valueToIndex_.contains(q));
  const auto index = valueToIndex_.lookup(q);

  valueToIndex_.erase(q);
  indexToValue_.erase(index);
}

TypedValue<QubitType> Qubits::getQubit(std::size_t index) const {
  assert(indexToValue_.contains(index));
  return indexToValue_.lookup(index);
}

std::size_t Qubits::getIndex(TypedValue<QubitType> q) const {
  assert(valueToIndex_.contains(q));
  return valueToIndex_.lookup(q);
}

void walkUnit(Region& region, WalkUnitFn fn) {
  Qubits qubits;
  for (Operation& curr : region.getOps()) {
    if (fn(&curr, qubits).wasInterrupted()) {
      break;
    };

    TypeSwitch<Operation*>(&curr)
        .Case<StaticOp>(
            [&](StaticOp op) { qubits.add(op.getQubit(), op.getIndex()); })
        .Case<AllocOp>([&](AllocOp op) { qubits.add(op.getResult()); })
        .Case<ExtractOp>([&](ExtractOp op) { qubits.add(op.getResult()); })
        .Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
          qubits.remap(op, WalkDirection::Forward);
        })
        .Case<ResetOp>([&](ResetOp op) {
          qubits.remap(op.getQubitIn(), op.getQubitOut(),
                       WalkDirection::Forward);
        })
        .Case<MeasureOp>([&](MeasureOp op) {
          qubits.remap(op.getQubitIn(), op.getQubitOut(),
                       WalkDirection::Forward);
        })
        .Case<InsertOp>([&](InsertOp op) { qubits.remove(op.getScalar()); })
        .Case<SinkOp>([&](SinkOp op) { qubits.remove(op.getQubit()); });
  }
}

LogicalResult walkCircuitGraph(MutableArrayRef<WireIterator> wires,
                               WalkDirection direction, WalkCircuitGraphFn fn) {
  const auto step = direction == WalkDirection::Forward ? 1 : -1;

  ReleasedOps released;

  PendingWiresMap pending;
  pending.reserve(wires.size());

  SmallVector<WireIterator*> curr;
  curr.reserve(wires.size());
  for (auto& it : wires) {
    curr.emplace_back(&it);
  }

  SmallVector<WireIterator*> next;
  next.reserve(wires.size());

  while (!curr.empty()) {
    for (WireIterator* itP : curr) {
      while (proceedOnWire(*itP, direction)) {
        const auto res =
            TypeSwitch<Operation*, CircuitWalkResult>(itP->operation())
                .Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
                  // If there are fewer wires than the qubit requires inputs,
                  // it's impossible to release the operation. Hence, fail.
                  if (op.getNumQubits() > wires.size()) {
                    return CircuitWalkResult::Fail;
                  }

                  if (op.getNumQubits() == 1) {
                    std::ranges::advance(*itP, step);
                    return CircuitWalkResult::Advance;
                  }

                  // Insert the unitary to the pending map.
                  // The caller decides if this op should be released.
                  const auto [it, inserted] = pending.try_emplace(op);
                  auto& wires = it->second;

                  if (inserted) {
                    wires.reserve(op.getNumQubits());
                  }

                  wires.emplace_back(itP);

                  return CircuitWalkResult::Hold; // Stop at multi-qubit gate.
                })
                .Case<AllocOp, StaticOp, ExtractOp, ResetOp, MeasureOp, SinkOp,
                      InsertOp>([&](auto) {
                  std::ranges::advance(*itP, step);
                  return CircuitWalkResult::Advance;
                })
                .Default([&](Operation* op) {
                  const auto name = op->getName().getStringRef();
                  report_fatal_error("unknown op encountered: " + name);
                  return CircuitWalkResult::Fail;
                });

        if (res == CircuitWalkResult::Hold) {
          break;
        }

        if (res == CircuitWalkResult::Fail) {
          return failure();
        }
      }
    }

    released.clear();
    const auto ready = make_filter_range(pending, IsReady{});
    const auto res = std::invoke(fn, ready, released);
    if (res.wasInterrupted() || res.wasSkipped()) {
      return failure();
    }

    for (UnitaryOpInterface op : released) {
      const auto& mapIt = pending.find(op);
      assert(mapIt != pending.end());

      for (auto& it : wires) {
        if (it.operation() == op) {
          std::ranges::advance(it, step);
          next.emplace_back(&it);
        }
      }

      pending.erase(mapIt);
    }

    curr = std::move(next);
    next.clear();
  }

  return success();
}
} // namespace mlir::qco
