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
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Drivers.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Value.h>

#include <cassert>
#include <cstddef>
#include <iterator>

namespace mlir::qco {

using namespace mlir::qtensor;

namespace {

using PendingWiresMap =
    DenseMap<UnitaryOpInterface, SmallVector<WireIterator*, 2>>;

/**
 * @brief Insert the unitary and the associated wire iterator into the pending
 * map.
 */
void insert(PendingWiresMap& map, UnitaryOpInterface op, WireIterator* wire) {
  auto [it, inserted] = map.try_emplace(op);
  auto& wires = it->second;

  if (inserted) {
    wires.reserve(op.getNumQubits());
  }

  wires.emplace_back(wire);
}

/**
 * @returns true if the wire iterator has not reached the end (Forward) or the
 * start (Backward) of the wire.
 */
bool proceedOnWire(const WireIterator& it, WalkDirection direction) {
  if (direction == WalkDirection::Forward) {
    return it != std::default_sentinel;
  }

  return !isa<AllocOp>(it.operation()) && !isa<StaticOp>(it.operation()) &&
         !isa<ExtractOp>(it.operation());
}
} // namespace

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

void walkQubitBlock(WireIterator& first, WireIterator& second,
                    WalkDirection direction) {
  const auto step = direction == WalkDirection::Forward ? 1 : -1;

  const auto advanceUntilTwoQubitOp = [&](WireIterator& it) {
    while (proceedOnWire(it, direction)) {
      if (auto op = dyn_cast<UnitaryOpInterface>(it.operation())) {
        if (op.getNumQubits() > 1) {
          break;
        }
      }

      std::ranges::advance(it, step);
    }
  };

  while (true) {
    advanceUntilTwoQubitOp(first);
    advanceUntilTwoQubitOp(second);

    if (!proceedOnWire(first, direction) || !proceedOnWire(second, direction)) {
      break;
    }

    if (first.operation() != second.operation()) {
      break;
    }

    std::ranges::advance(first, step);
    std::ranges::advance(second, step);
  }
}

LogicalResult walkCircuitGraph(MutableArrayRef<WireIterator> wires,
                               WalkDirection direction, WalkCircuitGraphFn fn) {
  const auto step = direction == WalkDirection::Forward ? 1 : -1;

  ReleasedIterators released;
  PendingWiresMap pending;
  SmallVector<ArrayRef<WireIterator*>> front;

  pending.reserve(wires.size());
  front.reserve((wires.size() + 1) / 2);

  while (true) {
    for (WireIterator& it : wires) {
      while (proceedOnWire(it, direction)) {
        const auto res =
            TypeSwitch<Operation*, WalkResult>(it.operation())
                .Case<BarrierOp>([&](BarrierOp op) {
                  insert(pending, op, &it);

                  // Release barrier directly.
                  if (pending[op].size() == op.getNumQubits()) {
                    for (WireIterator* wire : pending[op]) {
                      std::ranges::advance(*wire, step);
                    }
                    return WalkResult::advance();
                  }

                  return WalkResult::interrupt();
                })
                .Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
                  assert(op.getNumQubits() > 0 && op.getNumQubits() <= 2);

                  if (op.getNumQubits() == 1) {
                    std::ranges::advance(it, step);
                    return WalkResult::advance();
                  }

                  insert(pending, op, &it);
                  if (pending[op].size() == op.getNumQubits()) {
                    front.emplace_back(pending[op]);
                  }

                  return WalkResult::interrupt(); // Stop at two-qubit gate.
                })
                .Case<AllocOp, StaticOp, ResetOp, MeasureOp, SinkOp, ExtractOp,
                      InsertOp>([&](auto) {
                  std::ranges::advance(it, step);
                  return WalkResult::advance();
                })
                .Default([&](Operation* op) {
                  const auto name = op->getName().getStringRef();
                  report_fatal_error("unknown op encountered: " + name);
                  return WalkResult::interrupt();
                });

        if (res.wasInterrupted()) {
          break;
        }
      }
    }

    if (all_of(wires, [&](const WireIterator& it) {
          return !proceedOnWire(it, direction);
        })) {
      break;
    }

    released.clear();
    const auto res = std::invoke(fn, front, released);
    if (res.wasInterrupted() || res.wasSkipped()) {
      return failure();
    }

    for (WireIterator* it : released) {
      std::ranges::advance(*it, step);
    }

    front.clear();
    pending.clear();
  }

  return success();
}
} // namespace mlir::qco
