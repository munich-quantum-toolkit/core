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
#include <llvm/ADT/SmallVector.h>

#include <cstddef>

namespace mlir::qco::decomposition {

/**
 * Sequence of abstract decomposition gates plus a residual global phase.
 *
 * `gates` is stored in execution order: for a column state vector, the first
 * gate in the vector is applied first. The reconstructed 4x4 unitary
 * is therefore `U = e^{i * phi} * M_{n-1} * ... * M_0`, where `M_i` is the
 * two-qubit matrix for `gates[i]` and `phi` is `globalPhase` in radians (via
 * `helpers::globalPhaseFactor`).
 */
struct QubitGateSequence {
  /// Expected short decomposition length; `SmallVector` inline storage size.
  static constexpr unsigned GATES_INLINE_CAPACITY = 8;

  /// Gates in execution order (see struct comment).
  llvm::SmallVector<Gate, GATES_INLINE_CAPACITY> gates;

  /// Residual global phase in radians, not represented by explicit gates.
  double globalPhase{};

  /// True when `std::abs(globalPhase)` exceeds `DEFAULT_ATOL` in
  /// `EulerBasis.h`.
  [[nodiscard]] bool hasGlobalPhase() const;

  /// Heuristic complexity from `helpers::getComplexity()` for each gate, plus a
  /// synthetic global-phase term when `hasGlobalPhase()` is true.
  [[nodiscard]] std::size_t complexity() const;

  /**
   * Reconstruct the overall two-qubit unitary represented by the sequence.
   *
   * Single-qubit gates are expanded to the two-qubit workspace convention used
   * throughout the decomposition utilities.
   */
  [[nodiscard]] Eigen::Matrix4cd getUnitaryMatrix() const;
};

/// Documents intent only; same type as `QubitGateSequence`.
using OneQubitGateSequence = QubitGateSequence;
/// Documents intent only; same type as `QubitGateSequence`.
using TwoQubitGateSequence = QubitGateSequence;

} // namespace mlir::qco::decomposition
