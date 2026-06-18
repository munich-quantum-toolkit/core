/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/Helpers.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>

using namespace mlir::qco;
using namespace mlir::qco::helpers;

TEST(DecompositionHelpersTest, RemEuclidNeverNegative) {
  EXPECT_DOUBLE_EQ(remEuclid(-1.0, 3.0), 2.0);
  EXPECT_DOUBLE_EQ(remEuclid(7.0, 3.0), 1.0);
  EXPECT_DOUBLE_EQ(remEuclid(0.0, 2.5), 0.0);
}

TEST(DecompositionHelpersTest, TraceToFidelityMatchesFormula) {
  const std::complex<double> x{3.0, 4.0};
  const double absx = 5.0;
  EXPECT_DOUBLE_EQ(traceToFidelity(x), (4.0 + (absx * absx)) / 20.0);
}

TEST(DecompositionHelpersTest, GlobalPhaseFactorUnitMagnitude) {
  const auto z = globalPhaseFactor(1.25);
  EXPECT_NEAR(std::abs(z), 1.0, 1e-14);
}

TEST(DecompositionHelpersTest, IsUnitaryMatrixRejectsNonUnitary) {
  const Matrix2x2 m = Matrix2x2::fromElements(2.0, 0.0, 0.0, 2.0);
  EXPECT_FALSE(isUnitaryMatrix(m));
}

TEST(DecompositionHelpersTest, IsUnitaryMatrixAcceptsUnitary) {
  const Matrix2x2 m = Matrix2x2::identity();
  EXPECT_TRUE(isUnitaryMatrix(m));
}
