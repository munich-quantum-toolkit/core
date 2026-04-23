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
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  MatrixType randomMatrix;
  for (auto& x : randomMatrix.reshaped()) {
    x = std::complex<double>(dist(rng), dist(rng));
  }
  Eigen::HouseholderQR<MatrixType> qr{};
  qr.compute(randomMatrix);
  const MatrixType unitaryMatrix = qr.householderQ();
  assert(helpers::isUnitaryMatrix(unitaryMatrix));
  return unitaryMatrix;
}

} // namespace mlir::qco::decomposition_test
