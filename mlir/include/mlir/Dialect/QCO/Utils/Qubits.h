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

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace mlir::qco {

class Qubits {
  /// Specifies the qubit "location" (hardware or program).
  enum class QubitLocation : std::uint8_t { Hardware, Program };

public:
  /// Update the qubits map based on the given op.
  void update(Operation*);
  /// Add qubit with automatically assigned dynamic index.
  void add(TypedValue<QubitType> q);
  /// Add qubit with static index.
  void add(TypedValue<QubitType> q, std::size_t hw);
  /// Remap the qubit value from prev to next.
  void remap(TypedValue<QubitType> prev, TypedValue<QubitType> next);
  /// Remove the qubit value.
  void remove(TypedValue<QubitType> q);
  /// Return the qubit value assigned to a program index.
  [[nodiscard]] TypedValue<QubitType> getProgramQubit(std::size_t index) const;
  /// Return the qubit value assigned to a hardware index.
  [[nodiscard]] TypedValue<QubitType> getHardwareQubit(std::size_t index) const;
  /// Return the index assigned to the qubit value.
  [[nodiscard]] std::size_t getIndex(TypedValue<QubitType> q) const;

private:
  DenseMap<std::size_t, TypedValue<QubitType>> programToValue_;
  DenseMap<std::size_t, TypedValue<QubitType>> hardwareToValue_;
  DenseMap<TypedValue<QubitType>, std::pair<QubitLocation, std::size_t>>
      valueToIndex_;
};
} // namespace mlir::qco
