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

using PendingWiresMap =
    DenseMap<UnitaryOpInterface, SmallVector<WireIterator*, 2>>;

/**
 * @brief Insert the unitary and the associated wire iterator into the pending
 * map.
 */
static void insert(PendingWiresMap& map, UnitaryOpInterface op,
                   WireIterator* wire) {
  auto [it, inserted] = map.try_emplace(op);
  auto& wires = it->second;

  if (inserted) {
    wires.reserve(op.getNumQubits());
  }

  wires.emplace_back(wire);
}

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

  ReleasedIterators released;
  PendingWiresMap pending;

  SmallVector<SmallVector<WireIterator*>> front;

  pending.reserve(wires.size());
  front.reserve((wires.size() + 1) / 2);

  while (true) {
    for (WireIterator& it : wires) {
      while (proceedOnWire(it, direction)) {
        const auto res =
            TypeSwitch<Operation*, WalkResult>(it.operation())
                .Case<BarrierOp>([&](BarrierOp op) {
                  // If there are fewer wires than the qubit requires inputs,
                  // it's impossible to release the operation. Hence, fail.
                  if (op.getNumQubits() > wires.size()) {
                    return WalkResult::skip();
                  }

                  // Insert the barrier to the pending map.
                  // Release barrier directly.
                  insert(pending, op, &it);
                  if (pending[op].size() == op.getNumQubits()) {
                    for (WireIterator* wire : pending[op]) {
                      std::ranges::advance(*wire, step);
                    }
                    pending.erase(op);
                    return WalkResult::advance();
                  }

                  return WalkResult::interrupt();
                })
                .Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
                  assert(op.getNumQubits() > 0 && op.getNumQubits() <= 2);

                  // If there are fewer wires than the qubit requires inputs,
                  // it's impossible to release the operation. Hence, fail.
                  if (op.getNumQubits() > wires.size()) {
                    return WalkResult::skip();
                  }

                  if (op.getNumQubits() == 1) {
                    std::ranges::advance(it, step);
                    return WalkResult::advance();
                  }

                  // Insert the unitary to the pending map.
                  // The caller decides if this op should be released.
                  insert(pending, op, &it);
                  if (pending[op].size() == op.getNumQubits()) {
                    // Because pending may grow in size and invalidate keys
                    // and values, we need to copy pending here.
                    front.emplace_back(SmallVector(pending[op]));
                    pending.erase(op);
                  }

                  return WalkResult::interrupt(); // Stop at two-qubit gate.
                })
                .Case<AllocOp, StaticOp, ExtractOp, ResetOp, MeasureOp, SinkOp,
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

        if (res.wasSkipped()) {
          return failure();
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
      assert(isa<UnitaryOpInterface>(it->operation()));
      assert(!isa<BarrierOp>(it->operation()));

      std::ranges::advance(*it, step);
    }

    front.clear();
    pending.clear();
  }

  return success();
}
} // namespace mlir::qco
