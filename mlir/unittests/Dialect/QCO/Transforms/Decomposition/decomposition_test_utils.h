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

#include "mlir/Dialect/QCO/Transforms/Decomposition/Helpers.h"

#include <Eigen/Core>
#include <Eigen/QR>

#include <cassert>
#include <cmath>
#include <complex>
#include <random>

/// Standard U3 matrix (same convention as QCO ``u`` angles).
[[nodiscard]] inline Eigen::Matrix2cd u3Matrix(double theta, double phi,
                                               double lambda) {
  using Complex = std::complex<double>;
  const Complex i(0.0, 1.0);
  const double c = std::cos(theta / 2.0);
  const double s = std::sin(theta / 2.0);
  const Complex eiphi = std::exp(i * phi);
  const Complex eilambda = std::exp(i * lambda);
  const Complex eiphilambda = std::exp(i * (phi + lambda));

  Eigen::Matrix2cd mat;
  mat << c, -eilambda * s, eiphi * s, eiphilambda * c;
  return mat;
}

/// Compare up to a single global phase factor.
template <typename Matrix>
[[nodiscard]] bool isEquivalentUpToGlobalPhase(const Matrix& lhs,
                                               const Matrix& rhs,
                                               double atol = 1e-10) {
  const auto overlap = (rhs.adjoint() * lhs).trace();
  if (std::abs(overlap) <= atol) {
    return false;
  }
  const auto factor = overlap / std::abs(overlap);
  return lhs.isApprox(factor * rhs, atol);
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
  assert(mlir::qco::helpers::isUnitaryMatrix(unitaryMatrix));
  return unitaryMatrix;
}
