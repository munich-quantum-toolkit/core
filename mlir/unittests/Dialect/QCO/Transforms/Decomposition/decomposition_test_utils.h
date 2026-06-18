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

#include "TestCaseUtils.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Helpers.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/UnitaryMatrices.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <vector>

namespace mlir::qco::decomposition_test {

using mqt::test::isEquivalentUpToGlobalPhase;

/// Standard `U3(theta, phi, lambda)` matrix. Thin wrapper over the library
/// `uMatrix` so every test uses the same implementation.
[[nodiscard]] inline Matrix2x2 u3Matrix(double theta, double phi,
                                        double lambda) {
  return decomposition::uMatrix(theta, phi, lambda);
}

namespace detail {

/// Generate a Haar-ish random unitary as a row-major `dim x dim` buffer via
/// modified Gram-Schmidt on Gaussian-random complex columns.
[[nodiscard]] inline std::vector<std::complex<double>>
randomUnitaryData(std::size_t dim, std::mt19937& rng) {
  std::normal_distribution<double> normalDist(0.0, 1.0);
  std::vector<std::vector<std::complex<double>>> columns(
      dim, std::vector<std::complex<double>>(dim));
  for (auto& column : columns) {
    for (auto& entry : column) {
      entry = std::complex<double>(normalDist(rng), normalDist(rng));
    }
  }
  for (std::size_t j = 0; j < dim; ++j) {
    for (std::size_t k = 0; k < j; ++k) {
      std::complex<double> projection{0.0, 0.0};
      for (std::size_t i = 0; i < dim; ++i) {
        projection += std::conj(columns[k][i]) * columns[j][i];
      }
      for (std::size_t i = 0; i < dim; ++i) {
        columns[j][i] -= projection * columns[k][i];
      }
    }
    double norm = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      norm += std::norm(columns[j][i]);
    }
    norm = std::sqrt(norm);
    for (std::size_t i = 0; i < dim; ++i) {
      columns[j][i] /= norm;
    }
  }
  std::vector<std::complex<double>> data(dim * dim);
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      data[(row * dim) + col] = columns[col][row];
    }
  }
  return data;
}

} // namespace detail

/// Random `2×2` unitary matrix.
[[nodiscard]] inline Matrix2x2 randomUnitary2x2(std::mt19937& rng) {
  const auto data = detail::randomUnitaryData(2, rng);
  const Matrix2x2 unitary =
      Matrix2x2::fromElements(data[0], data[1], data[2], data[3]);
  assert(helpers::isUnitaryMatrix(unitary));
  return unitary;
}

/// Random `4×4` unitary matrix.
[[nodiscard]] inline Matrix4x4 randomUnitary4x4(std::mt19937& rng) {
  const auto data = detail::randomUnitaryData(4, rng);
  const Matrix4x4 unitary = Matrix4x4::fromElements(
      data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
      data[8], data[9], data[10], data[11], data[12], data[13], data[14],
      data[15]);
  assert(helpers::isUnitaryMatrix(unitary));
  return unitary;
}

} // namespace mlir::qco::decomposition_test
