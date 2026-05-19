/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

using namespace mlir::qco::helpers;

TEST(EulerHelpersTest, Mod2piWrapsIntoHalfOpenInterval) {
  EXPECT_NEAR(mod2pi(0.0), 0.0, 1e-14);
  EXPECT_NEAR(mod2pi(std::numbers::pi), -std::numbers::pi, 1e-12);
  EXPECT_NEAR(mod2pi(3.0 * std::numbers::pi), -std::numbers::pi, 1e-12);
}

TEST(EulerHelpersTest, IsUnitaryMatrixRejectsNonUnitary) {
  Eigen::Matrix2cd m;
  m << 2.0, 0.0, 0.0, 2.0;
  EXPECT_FALSE(isUnitaryMatrix(m));
}

TEST(EulerHelpersTest, IsUnitaryMatrixAcceptsUnitary) {
  const Eigen::Matrix2cd m = Eigen::Matrix2cd::Identity();
  EXPECT_TRUE(isUnitaryMatrix(m));
}
