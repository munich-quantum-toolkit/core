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

#include <Eigen/Core>
#include <Eigen/QR>

#include <cassert>
#include <complex>
#include <random>

namespace mlir::qco::decomposition_test {

using mqt::test::isEquivalentUpToGlobalPhase;

/// Standard `U3(theta, phi, lambda)` matrix. Thin wrapper over the library
/// `uMatrix` so every test uses the same implementation.
[[nodiscard]] inline Eigen::Matrix2cd u3Matrix(double theta, double phi,
                                               double lambda) {
  return decomposition::uMatrix(theta, phi, lambda);
}

template <typename MatrixType>
[[nodiscard]] MatrixType randomUnitaryMatrix(std::mt19937& rng) {
  static_assert(MatrixType::RowsAtCompileTime != Eigen::Dynamic &&
                    MatrixType::ColsAtCompileTime != Eigen::Dynamic,
                "randomUnitaryMatrix requires fixed-size matrices");
  std::normal_distribution<double> normalDist(0.0, 1.0);
  MatrixType randomMatrix;
  for (auto& x : randomMatrix.reshaped()) {
    x = std::complex<double>(normalDist(rng), normalDist(rng));
  }
  Eigen::HouseholderQR<MatrixType> qr{};
  qr.compute(randomMatrix);
  const MatrixType qMatrix = qr.householderQ();
  const MatrixType rMatrix =
      qr.matrixQR().template triangularView<Eigen::Upper>();
  MatrixType dMatrix = MatrixType::Identity();
  constexpr Eigen::Index dim = MatrixType::RowsAtCompileTime;
  for (Eigen::Index i = 0; i < dim; ++i) {
    const auto rii = rMatrix(i, i);
    const auto absRii = std::abs(rii);
    dMatrix(i, i) =
        absRii > 0.0 ? (rii / absRii) : std::complex<double>{1.0, 0.0};
  }
  const MatrixType unitaryMatrix = qMatrix * dMatrix;
  assert(helpers::isUnitaryMatrix(unitaryMatrix));
  return unitaryMatrix;
}

} // namespace mlir::qco::decomposition_test
