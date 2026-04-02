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
#include "mlir/Dialect/QCO/Utils/Drivers.h"

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

std::size_t Qubits::getHardwareIndex(TypedValue<QubitType> q) const {
  assert(valueToIndex_.contains(q));
  const auto& [location, index] = valueToIndex_.lookup(q);
  assert(location == QubitLocation::Hardware);
  return index;
}

namespace impl {
std::optional<ArrayRef<WireIterator*>>
tryReleaseReadyWires(PendingWiresMap& map, UnitaryOpInterface op,
                     WireIterator* wire) {
  auto [it, inserted] = map.try_emplace(op);
  auto& wires = it->second;

  if (inserted) {
    wires.reserve(op.getNumQubits());
  }

  wires.emplace_back(wire);

  if (wires.size() == op.getNumQubits()) {
    return std::move(wires);
  }

  return std::nullopt;
}
} // namespace impl
} // namespace mlir::qco
