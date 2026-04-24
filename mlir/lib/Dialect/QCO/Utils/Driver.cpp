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

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Value.h>

#include <cassert>
#include <cstddef>
#include <utility>

namespace mlir::qco {
void Qubits::add(TypedValue<QubitType> q) {
  const auto index = programToValue_.size();
  programToValue_.try_emplace(index, q);
  valueToIndex_.try_emplace(q, std::make_pair(QubitLocation::Program, index));
}

void Qubits::add(TypedValue<QubitType> q, std::size_t hw) {
  hardwareToValue_.try_emplace(hw, q);
  valueToIndex_.try_emplace(q, std::make_pair(QubitLocation::Hardware, hw));
}

void Qubits::remap(TypedValue<QubitType> prev, TypedValue<QubitType> next) {
  assert(valueToIndex_.contains(prev));
  const auto& [location, index] = valueToIndex_.lookup(prev);

  valueToIndex_.erase(prev);
  valueToIndex_.try_emplace(next, std::make_pair(location, index));

  if (location == QubitLocation::Program) {
    programToValue_[index] = next;
    return;
  }

  hardwareToValue_[index] = next;
}

void Qubits::remove(TypedValue<QubitType> q) {
  assert(valueToIndex_.contains(q));
  const auto& [location, index] = valueToIndex_.lookup(q);

  valueToIndex_.erase(q);

  if (location == QubitLocation::Program) {
    programToValue_.erase(index);
    return;
  }

  hardwareToValue_.erase(index);
}

TypedValue<QubitType> Qubits::getProgramQubit(std::size_t index) const {
  assert(programToValue_.contains(index));
  return programToValue_.lookup(index);
}

TypedValue<QubitType> Qubits::getHardwareQubit(std::size_t index) const {
  assert(hardwareToValue_.contains(index));
  return hardwareToValue_.lookup(index);
}

void walkProgram(Region& region, WalkProgramFn fn) {
  Qubits qubits;
  for (Operation& curr : region.getOps()) {
    if (fn(&curr, qubits).wasInterrupted()) {
      break;
    };

    TypeSwitch<Operation*>(&curr)
        .template Case<StaticOp>(
            [&](StaticOp op) { qubits.add(op.getQubit(), op.getIndex()); })
        .template Case<AllocOp>([&](AllocOp op) { qubits.add(op.getResult()); })
        .template Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
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
  }
}
} // namespace mlir::qco
