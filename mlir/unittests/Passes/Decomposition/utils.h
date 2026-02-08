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

#include "mlir/Passes/Decomposition/Helpers.h"

#include <Eigen/QR>
#include <cassert>
#include <complex>
#include <random>

template <typename MatrixType> [[nodiscard]] MatrixType randomUnitaryMatrix() {
  [[maybe_unused]] static auto rng = []() { return std::mt19937{123456UL}; }();
  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  MatrixType randomMatrix;
  for (auto& x : randomMatrix.reshaped()) {
    x = std::complex<double>(dist(rng), dist(rng));
  }
  Eigen::HouseholderQR<MatrixType> qr{}; // NOLINT(misc-include-cleaner)
  qr.compute(randomMatrix);
  const MatrixType unitaryMatrix = qr.householderQ();
  assert(mlir::qco::helpers::isUnitaryMatrix(unitaryMatrix));
  return unitaryMatrix;
}
