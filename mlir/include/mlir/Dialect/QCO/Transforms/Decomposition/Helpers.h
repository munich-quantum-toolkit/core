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

#include "mlir/Dialect/QCO/Transforms/Decomposition/GateKind.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <complex>

/// Numeric + classification helpers used by the decomposition passes.
/// Lives in `mlir::qco::helpers` (not `decomposition`) because some helpers
/// map IR ops back to decomposition kinds.

namespace mlir::qco::helpers {

/// Check whether `matrix` is unitary within `tolerance` (i.e. `M^H M` is
/// approximately `I`, using Eigen's `isIdentity`).
template <typename T, int N, int M>
[[nodiscard]] bool isUnitaryMatrix(const Eigen::Matrix<T, N, M>& matrix,
                                   double tolerance = 1e-12) {
  if (matrix.rows() != matrix.cols()) {
    return false;
  }
  return (matrix.adjoint() * matrix).isIdentity(tolerance);
}
// NOLINTEND(misc-include-cleaner)

/**
 * Euclidean remainder of a modulo b.
 * The returned value is never negative.
 */
[[nodiscard]] double remEuclid(double a, double b);

/**
 * Wrap angle into interval [-pi, pi). If within atol of the endpoint, clamp to
 * -pi.
 */
[[nodiscard]] double mod2pi(double angle, double angleZeroEpsilon = 1e-13);

/**
 * Return the heuristic cost assigned to a gate acting on `numOfQubits`.
 */
[[nodiscard]] std::size_t getComplexity(decomposition::GateKind type,
                                        std::size_t numOfQubits);

/**
 * Return the scalar `e^(i * globalPhase)` factor for a stored global phase.
 */
[[nodiscard]] std::complex<double> globalPhaseFactor(double globalPhase);

} // namespace mlir::qco::helpers
