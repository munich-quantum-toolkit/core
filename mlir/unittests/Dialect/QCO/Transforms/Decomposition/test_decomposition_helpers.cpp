/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/GateKind.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Helpers.h"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <numbers>

using namespace mlir::qco::helpers;
using namespace mlir::qco::decomposition;

TEST(DecompositionHelpersTest, RemEuclidNeverNegative) {
  EXPECT_DOUBLE_EQ(remEuclid(-1.0, 3.0), 2.0);
  EXPECT_DOUBLE_EQ(remEuclid(7.0, 3.0), 1.0);
  EXPECT_DOUBLE_EQ(remEuclid(0.0, 2.5), 0.0);
}

TEST(DecompositionHelpersTest, Mod2piWrapsIntoHalfOpenInterval) {
  EXPECT_NEAR(mod2pi(0.0), 0.0, 1e-14);
  EXPECT_NEAR(mod2pi(std::numbers::pi), -std::numbers::pi, 1e-12);
  EXPECT_NEAR(mod2pi(3.0 * std::numbers::pi), -std::numbers::pi, 1e-12);
}

TEST(DecompositionHelpersTest, TraceToFidelityMatchesFormula) {
  const std::complex<double> x{3.0, 4.0};
  const double absx = 5.0;
  EXPECT_DOUBLE_EQ(traceToFidelity(x), (4.0 + (absx * absx)) / 20.0);
}

TEST(DecompositionHelpersTest, GetComplexitySingleQubitAndGphase) {
  EXPECT_EQ(getComplexity(GateKind::X, 1), 1U);
  EXPECT_EQ(getComplexity(GateKind::GPhase, 1), 0U);
}

TEST(DecompositionHelpersTest, GetComplexityMultiQubitUsesFactorModel) {
  EXPECT_EQ(getComplexity(GateKind::RZZ, 2), 10U);
}

TEST(DecompositionHelpersTest, GlobalPhaseFactorUnitMagnitude) {
  const auto z = globalPhaseFactor(1.25);
  EXPECT_NEAR(std::abs(z), 1.0, 1e-14);
}

TEST(DecompositionHelpersTest, IsUnitaryMatrixRejectsNonUnitary) {
  Eigen::Matrix2cd m;
  m << 2.0, 0.0, 0.0, 2.0;
  EXPECT_FALSE(isUnitaryMatrix(m));
}

TEST(DecompositionHelpersTest, IsUnitaryMatrixAcceptsUnitary) {
  const Eigen::Matrix2cd m = Eigen::Matrix2cd::Identity();
  EXPECT_TRUE(isUnitaryMatrix(m));
}
