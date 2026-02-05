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
#include "mlir/Passes/Decomposition/UnitaryMatrices.h"

#include <Eigen/Core>
#include <cmath>
#include <gtest/gtest.h>

using namespace mlir::qco;
using namespace mlir::qco::helpers;
using namespace mlir::qco::decomposition;

TEST(HelpersTest, RemEuclidPositiveValues) {
  EXPECT_NEAR(remEuclid(5.0, 3.0), 2.0, 1e-12);
  EXPECT_NEAR(remEuclid(7.5, 2.5), 0.0, 1e-12);
  EXPECT_NEAR(remEuclid(10.0, 4.0), 2.0, 1e-12);
}

TEST(HelpersTest, RemEuclidNegativeValues) {
  EXPECT_NEAR(remEuclid(-5.0, 3.0), 1.0, 1e-12);
  EXPECT_NEAR(remEuclid(-7.0, 4.0), 1.0, 1e-12);
  EXPECT_NEAR(remEuclid(-1.0, 5.0), 4.0, 1e-12);
}

TEST(HelpersTest, RemEuclidZero) {
  EXPECT_NEAR(remEuclid(0.0, 5.0), 0.0, 1e-12);
}

TEST(HelpersTest, Mod2piPositiveAngles) {
  EXPECT_NEAR(mod2pi(0.0), 0.0, 1e-12);
  EXPECT_NEAR(mod2pi(qc::PI), -qc::PI, 1e-12);
  EXPECT_NEAR(mod2pi(qc::PI_2), qc::PI_2, 1e-12);
  EXPECT_NEAR(mod2pi(qc::TAU), -qc::PI, 1e-12);
}

TEST(HelpersTest, Mod2piNegativeAngles) {
  EXPECT_NEAR(mod2pi(-qc::PI_2), -qc::PI_2, 1e-12);
  EXPECT_NEAR(mod2pi(-qc::PI), -qc::PI, 1e-12);
  EXPECT_NEAR(mod2pi(-qc::TAU), -qc::PI, 1e-12);
}

TEST(HelpersTest, Mod2piLargeAngles) {
  EXPECT_NEAR(mod2pi(3.0 * qc::PI), -qc::PI, 1e-12);
  EXPECT_NEAR(mod2pi(5.0 * qc::PI), -qc::PI, 1e-12);
  EXPECT_NEAR(mod2pi(-3.0 * qc::PI), -qc::PI, 1e-12);
}

TEST(HelpersTest, TraceToFidelityMaximal) {
  // |trace| = 4 should give fidelity = 1
  qfp trace(4.0, 0.0);
  EXPECT_NEAR(traceToFidelity(trace), 1.0, 1e-12);
}

TEST(HelpersTest, TraceToFidelityZero) {
  // |trace| = 0 should give fidelity = 0.2
  qfp trace(0.0, 0.0);
  EXPECT_NEAR(traceToFidelity(trace), 0.2, 1e-12);
}

TEST(HelpersTest, TraceToFidelityIntermediate) {
  // Test intermediate values
  qfp trace(2.0, 0.0);
  fp fidelity = traceToFidelity(trace);
  EXPECT_GT(fidelity, 0.2);
  EXPECT_LT(fidelity, 1.0);
}

TEST(HelpersTest, TraceToFidelityComplex) {
  // Test complex trace
  qfp trace(3.0, 1.0);
  fp fidelity = traceToFidelity(trace);
  EXPECT_GT(fidelity, 0.2);
  EXPECT_LE(fidelity, 1.0);
}

TEST(HelpersTest, GetComplexitySingleQubitGates) {
  EXPECT_EQ(getComplexity(qc::X, 1), 1UL);
  EXPECT_EQ(getComplexity(qc::Y, 1), 1UL);
  EXPECT_EQ(getComplexity(qc::Z, 1), 1UL);
  EXPECT_EQ(getComplexity(qc::H, 1), 1UL);
  EXPECT_EQ(getComplexity(qc::RX, 1), 1UL);
  EXPECT_EQ(getComplexity(qc::RY, 1), 1UL);
  EXPECT_EQ(getComplexity(qc::RZ, 1), 1UL);
}

TEST(HelpersTest, GetComplexityTwoQubitGates) {
  EXPECT_EQ(getComplexity(qc::X, 2), 10UL);
  EXPECT_EQ(getComplexity(qc::RXX, 2), 10UL);
  EXPECT_EQ(getComplexity(qc::RYY, 2), 10UL);
  EXPECT_EQ(getComplexity(qc::RZZ, 2), 10UL);
}

TEST(HelpersTest, GetComplexityMultiQubitGates) {
  EXPECT_EQ(getComplexity(qc::X, 3), 20UL);
  EXPECT_EQ(getComplexity(qc::X, 4), 30UL);
  EXPECT_EQ(getComplexity(qc::X, 5), 40UL);
}

TEST(HelpersTest, GetComplexityGlobalPhase) {
  EXPECT_EQ(getComplexity(qc::GPhase, 0), 0UL);
}

TEST(HelpersTest, KroneckerProductIdentity) {
  auto result = kroneckerProduct(IDENTITY_GATE, IDENTITY_GATE);
  matrix4x4 expected = matrix4x4::Identity();
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(HelpersTest, KroneckerProductPauliX) {
  matrix2x2 pauliX{{0, 1}, {1, 0}};
  auto result = kroneckerProduct(pauliX, IDENTITY_GATE);

  matrix4x4 expected{{0, 0, 1, 0}, {0, 0, 0, 1}, {1, 0, 0, 0}, {0, 1, 0, 0}};

  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(HelpersTest, KroneckerProductNonCommutative) {
  matrix2x2 pauliX{{0, 1}, {1, 0}};
  matrix2x2 pauliZ{{1, 0}, {0, -1}};

  auto resultXZ = kroneckerProduct(pauliX, pauliZ);
  auto resultZX = kroneckerProduct(pauliZ, pauliX);

  // X ⊗ Z should not equal Z ⊗ X
  EXPECT_FALSE(resultXZ.isApprox(resultZX, 1e-12));
}

TEST(HelpersTest, IsUnitaryMatrixIdentity2x2) {
  EXPECT_TRUE(isUnitaryMatrix(IDENTITY_GATE));
}

TEST(HelpersTest, IsUnitaryMatrixIdentity4x4) {
  matrix4x4 identity = matrix4x4::Identity();
  EXPECT_TRUE(isUnitaryMatrix(identity));
}

TEST(HelpersTest, IsUnitaryMatrixRotations) {
  EXPECT_TRUE(isUnitaryMatrix(rxMatrix(1.5)));
  EXPECT_TRUE(isUnitaryMatrix(ryMatrix(2.3)));
  EXPECT_TRUE(isUnitaryMatrix(rzMatrix(0.7)));
}

TEST(HelpersTest, IsUnitaryMatrixNonUnitary) {
  matrix2x2 nonUnitary{{2, 0}, {0, 2}};
  EXPECT_FALSE(isUnitaryMatrix(nonUnitary));

  matrix2x2 nonUnitary2{{1, 1}, {0, 1}};
  EXPECT_FALSE(isUnitaryMatrix(nonUnitary2));
}

TEST(HelpersTest, IsUnitaryMatrixProductOfUnitaries) {
  auto product = rxMatrix(1.0) * ryMatrix(2.0) * rzMatrix(3.0);
  EXPECT_TRUE(isUnitaryMatrix(product));
}

TEST(HelpersTest, IsUnitaryMatrixWithGlobalPhase) {
  auto matrixWithPhase = std::exp(qfp(0, qc::PI / 4.0)) * IDENTITY_GATE;
  EXPECT_TRUE(isUnitaryMatrix(matrixWithPhase));
}

TEST(HelpersTest, SelfAdjointEvdIdentity) {
  matrix2x2 identity = matrix2x2::Identity();
  auto [vecs, vals] = selfAdjointEvd(identity);

  // Eigenvalues should be all 1s
  EXPECT_NEAR(vals(0), 1.0, 1e-12);
  EXPECT_NEAR(vals(1), 1.0, 1e-12);
}

TEST(HelpersTest, SelfAdjointEvdPauliZ) {
  matrix2x2 pauliZ{{1, 0}, {0, -1}};
  auto [vecs, vals] = selfAdjointEvd(pauliZ);

  // Eigenvalues should be +1 and -1
  EXPECT_NEAR(std::abs(vals(0)), 1.0, 1e-12);
  EXPECT_NEAR(std::abs(vals(1)), 1.0, 1e-12);
  EXPECT_NEAR(vals(0) + vals(1), 0.0, 1e-12);
}

TEST(HelpersTest, SelfAdjointEvdDiagonal) {
  matrix4x4 diagonal{{1, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 3, 0}, {0, 0, 0, 4}};
  auto [vecs, vals] = selfAdjointEvd(diagonal);

  // Eigenvalues should be 1, 2, 3, 4 (in some order)
  EXPECT_NEAR(vals(0) + vals(1) + vals(2) + vals(3), 10.0, 1e-12);
}