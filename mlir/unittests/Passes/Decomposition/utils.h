/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Passes/Decomposition/Helpers.h"

#include <Eigen/QR>
#include <cassert>
#include <cstdlib>
#include <gtest/gtest.h>
#include <unsupported/Eigen/KroneckerProduct>

template <typename MatrixType> [[nodiscard]] MatrixType randomUnitaryMatrix() {
  [[maybe_unused]] static auto initializeRandom = []() {
    // Eigen uses std::rand() internally, use fixed seed for deterministic
    // testing behavior
    std::srand(123456UL);
    return true;
  }();
  const MatrixType randomMatrix = MatrixType::Random();
  Eigen::HouseholderQR<MatrixType> qr{}; // NOLINT(misc-include-cleaner)
  qr.compute(randomMatrix);
  const MatrixType unitaryMatrix = qr.householderQ();
  assert(mlir::qco::helpers::isUnitaryMatrix(unitaryMatrix));
  return unitaryMatrix;
}
