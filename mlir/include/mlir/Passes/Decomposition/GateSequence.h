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

#include "Gate.h"

#include <Eigen/Core>
#include <cstddef>
#include <llvm/ADT/SmallVector.h>

namespace mlir::qco::decomposition {
/**
 * Gate sequence of single-qubit and/or two-qubit gates.
 */
struct QubitGateSequence {
  /**
   * Container sorting the gate sequence in order.
   */
  llvm::SmallVector<Gate, 8> gates;

  /**
   * Global phase adjustment required for the sequence.
   */
  double globalPhase{};
  /**
   * @return true if the global phase adjustment is not zero.
   */
  [[nodiscard]] bool hasGlobalPhase() const;

  /**
   * Calculate complexity of sequence according to getComplexity().
   */
  [[nodiscard]] std::size_t complexity() const;

  /**
   * Calculate overall unitary matrix of the sequence.
   */
  [[nodiscard]] Eigen::Matrix4cd getUnitaryMatrix() const;
};
/**
 * Helper type to show that a gate sequence is supposed to only contain
 * single-qubit gates.
 */
using OneQubitGateSequence = QubitGateSequence;
/**
 * Helper type to show that the gate sequence may contain two-qubit gates.
 */
using TwoQubitGateSequence = QubitGateSequence;

} // namespace mlir::qco::decomposition
