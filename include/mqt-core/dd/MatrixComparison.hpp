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

#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/mqt_core_dd_export.h"

#include <vector>

namespace dd {
/// Whether the normalized Frobenius distance meets the comparison tolerance.
struct IdentityComparison {
  bool closeToIdentity = false;
  bool closeUpToGlobalPhase = false;
};

/// Compare a matrix with identity, with and without one global phase.
/// Distances are divided by the square root of the matrix dimension. Root
/// weights are included; adding idle qubits leaves the distances unchanged.
/// A bounded lower-bound traversal rejects obvious failures before computing
/// the complete distance. Requires finite weights and IEEE arithmetic.
/// @param matrix The matrix DD, without ancillary input projection.
/// @param tolerance The nonnegative normalized Frobenius tolerance.
/// @return Both fixed-phase and phase-free comparisons with the tolerance.
[[nodiscard]] MQT_CORE_DD_EXPORT IdentityComparison
compareToIdentity(const mEdge& matrix, fp tolerance = 1e-10);

/// Test the normalized Frobenius weight outside measured-output diagonal
/// blocks. Entries whose row and column differ on a non-garbage qubit
/// contribute to the distance. All other entries are unrestricted. For a
/// relative unitary UV-dagger without ancillary inputs, zero distance means
/// equal measurement probabilities for every input. This is not a
/// restricted-input comparison.
/// @param matrix The matrix DD.
/// @param tolerance The nonnegative normalized Frobenius tolerance.
/// @param garbage Qubits ignored in the output comparison; missing entries are
/// measured qubits.
[[nodiscard]] MQT_CORE_DD_EXPORT bool
isCloseToDiagonal(const mEdge& matrix, fp tolerance = 1e-10,
                  const std::vector<bool>& garbage = {});
} // namespace dd
