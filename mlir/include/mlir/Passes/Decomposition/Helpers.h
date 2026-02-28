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

#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <complex>
#include <type_traits>

namespace mlir::qco::helpers {

[[nodiscard]] qc::OpType getQcType(UnitaryOpInterface op);

// NOLINTBEGIN(misc-include-cleaner)
template <typename T, int N, int M>
[[nodiscard]] auto selfAdjointEvd(const Eigen::Matrix<T, N, M>& a) {
  Eigen::SelfAdjointEigenSolver<std::remove_cvref_t<decltype(a)>> s;
  s.compute(a);
  auto vecs = s.eigenvectors().eval();
  auto vals = s.eigenvalues();
  return std::make_pair(vecs, vals);
}

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
 * Wrap angle into interval [-π,π). If within atol of the endpoint, clamp to -π.
 */
[[nodiscard]] double mod2pi(double angle, double angleZeroEpsilon = 1e-13);

/**
 * Calculate fidelity value of given trace.
 */
[[nodiscard]] double traceToFidelity(const std::complex<double>& x);

/**
 * Get complexity of gate operating on given number of qubits.
 */
[[nodiscard]] std::size_t getComplexity(qc::OpType type,
                                        std::size_t numOfQubits);

/**
 * Return complex factor which can be multiplied with the operation matrix.
 */
[[nodiscard]] std::complex<double> globalPhaseFactor(double globalPhase);

} // namespace mlir::qco::helpers
