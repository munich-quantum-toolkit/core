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

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateKind.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <complex>
#include <type_traits>

/// Numeric + classification helpers used by the decomposition passes.
/// Lives in `mlir::qco::helpers` (not `decomposition`) because some helpers
/// map IR ops back to decomposition kinds.

namespace mlir::qco::helpers {

/**
 * Map a QCO unitary operation to the corresponding decomposition `GateKind`.
 *
 * For controlled operations, this returns the wrapped body operation type
 * rather than the outer `ctrl` marker.
 */
[[nodiscard]] decomposition::GateKind getGateKind(UnitaryOpInterface op);

// NOLINTBEGIN(misc-include-cleaner)
/// Eigen-decomposition of a self-adjoint matrix. Returns `(eigenvectors,
/// eigenvalues)`; eigenvalues are real and sorted ascending.
template <typename T, int N, int M>
[[nodiscard]] auto selfAdjointEvd(const Eigen::Matrix<T, N, M>& a) {
  Eigen::SelfAdjointEigenSolver<std::remove_cvref_t<decltype(a)>> s;
  s.compute(a);
  auto vecs = s.eigenvectors().eval();
  auto vals = s.eigenvalues();
  return std::make_pair(vecs, vals);
}

/// Check whether `matrix` is unitary within `tolerance` (i.e. `M^H M` is
/// approximately `I`, using Eigen's `isIdentity`).
template <typename T, int N, int M>
[[nodiscard]] bool isUnitaryMatrix(const Eigen::Matrix<T, N, M>& matrix,
                                   double tolerance = 1e-12) {
  return (matrix.transpose().conjugate() * matrix).isIdentity(tolerance);
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
 * Convert a two-qubit trace overlap into the average gate fidelity metric used
 * by the decomposition cost code.
 */
[[nodiscard]] double traceToFidelity(const std::complex<double>& x);

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
