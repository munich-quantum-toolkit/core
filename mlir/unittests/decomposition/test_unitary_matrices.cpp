/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Passes/Decomposition/Gate.h"
#include "mlir/Passes/Decomposition/Helpers.h"
#include "mlir/Passes/Decomposition/UnitaryMatrices.h"

#include <Eigen/Core>
#include <cmath>
#include <gtest/gtest.h>

using namespace mlir::qco;
using namespace mlir::qco::decomposition;
using namespace mlir::qco::helpers;

TEST(UnitaryMatricesTest, RxMatrixZeroAngle) {
  auto result = rxMatrix(0.0);
  EXPECT_TRUE(result.isApprox(IDENTITY_GATE, 1e-12));
}

TEST(UnitaryMatricesTest, RxMatrixPi) {
  auto result = rxMatrix(qc::PI);
  matrix2x2 expected{{0, qfp(0, -1)}, {qfp(0, -1), 0}};
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, RxMatrixPiOver2) {
  auto result = rxMatrix(qc::PI_2);
  fp sqrtHalf = 1.0 / std::sqrt(2.0);
  matrix2x2 expected{{sqrtHalf, qfp(0, -sqrtHalf)},
                     {qfp(0, -sqrtHalf), sqrtHalf}};
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, RyMatrixZeroAngle) {
  auto result = ryMatrix(0.0);
  EXPECT_TRUE(result.isApprox(IDENTITY_GATE, 1e-12));
}

TEST(UnitaryMatricesTest, RyMatrixPi) {
  auto result = ryMatrix(qc::PI);
  matrix2x2 expected{{0, -1}, {1, 0}};
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, RyMatrixPiOver2) {
  auto result = ryMatrix(qc::PI_2);
  fp sqrtHalf = 1.0 / std::sqrt(2.0);
  matrix2x2 expected{{sqrtHalf, -sqrtHalf}, {sqrtHalf, sqrtHalf}};
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, RzMatrixZeroAngle) {
  auto result = rzMatrix(0.0);
  EXPECT_TRUE(result.isApprox(IDENTITY_GATE, 1e-12));
}

TEST(UnitaryMatricesTest, RzMatrixPi) {
  auto result = rzMatrix(qc::PI);
  matrix2x2 expected{{qfp(0, -1), 0}, {0, qfp(0, 1)}};
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, RzMatrixPiOver2) {
  auto result = rzMatrix(qc::PI_2);
  fp sqrtHalf = 1.0 / std::sqrt(2.0);
  matrix2x2 expected{{qfp(sqrtHalf, -sqrtHalf), 0},
                     {0, qfp(sqrtHalf, sqrtHalf)}};
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, RxxMatrixZeroAngle) {
  auto result = rxxMatrix(0.0);
  matrix4x4 expected = kroneckerProduct(IDENTITY_GATE, IDENTITY_GATE);
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, RxxMatrixPi) {
  auto result = rxxMatrix(qc::PI);
  EXPECT_TRUE(isUnitaryMatrix(result));
}

TEST(UnitaryMatricesTest, RyyMatrixZeroAngle) {
  auto result = ryyMatrix(0.0);
  matrix4x4 expected = kroneckerProduct(IDENTITY_GATE, IDENTITY_GATE);
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, RyyMatrixPi) {
  auto result = ryyMatrix(qc::PI);
  EXPECT_TRUE(isUnitaryMatrix(result));
}

TEST(UnitaryMatricesTest, RzzMatrixZeroAngle) {
  auto result = rzzMatrix(0.0);
  matrix4x4 expected = kroneckerProduct(IDENTITY_GATE, IDENTITY_GATE);
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, RzzMatrixPi) {
  auto result = rzzMatrix(qc::PI);
  EXPECT_TRUE(isUnitaryMatrix(result));
}

TEST(UnitaryMatricesTest, PMatrixZeroAngle) {
  auto result = pMatrix(0.0);
  EXPECT_TRUE(result.isApprox(IDENTITY_GATE, 1e-12));
}

TEST(UnitaryMatricesTest, PMatrixPi) {
  auto result = pMatrix(qc::PI);
  matrix2x2 expected{{1, 0}, {0, -1}};
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, UMatrixIdentity) {
  auto result = uMatrix(0.0, 0.0, 0.0);
  EXPECT_TRUE(result.isApprox(IDENTITY_GATE, 1e-12));
}

TEST(UnitaryMatricesTest, UMatrixIsUnitary) {
  auto result = uMatrix(1.5, 0.7, 2.3);
  EXPECT_TRUE(isUnitaryMatrix(result));
}

TEST(UnitaryMatricesTest, U2MatrixIsUnitary) {
  auto result = u2Matrix(1.5, 0.7);
  EXPECT_TRUE(isUnitaryMatrix(result));
}

TEST(UnitaryMatricesTest, HGateIsUnitary) {
  EXPECT_TRUE(isUnitaryMatrix(H_GATE));
}

TEST(UnitaryMatricesTest, HGateSquaredIsIdentity) {
  auto result = H_GATE * H_GATE;
  EXPECT_TRUE(result.isApprox(IDENTITY_GATE, 1e-12));
}

TEST(UnitaryMatricesTest, SwapGateIsUnitary) {
  EXPECT_TRUE(isUnitaryMatrix(SWAP_GATE));
}

TEST(UnitaryMatricesTest, SwapGateSquaredIsIdentity) {
  auto result = SWAP_GATE * SWAP_GATE;
  matrix4x4 expected = kroneckerProduct(IDENTITY_GATE, IDENTITY_GATE);
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, ExpandToTwoQubitsQubit0) {
  auto singleQubitGate = rxMatrix(1.5);
  auto result = expandToTwoQubits(singleQubitGate, 0);

  auto expected = kroneckerProduct(IDENTITY_GATE, singleQubitGate);
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, ExpandToTwoQubitsQubit1) {
  auto singleQubitGate = ryMatrix(2.3);
  auto result = expandToTwoQubits(singleQubitGate, 1);

  auto expected = kroneckerProduct(singleQubitGate, IDENTITY_GATE);
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, FixTwoQubitMatrixQubitOrder01) {
  auto originalMatrix = rxxMatrix(1.5);
  auto result = fixTwoQubitMatrixQubitOrder(originalMatrix, {0, 1});

  // Should be unchanged
  EXPECT_TRUE(result.isApprox(originalMatrix, 1e-12));
}

TEST(UnitaryMatricesTest, FixTwoQubitMatrixQubitOrder10) {
  auto originalMatrix = rxxMatrix(1.5);
  auto result = fixTwoQubitMatrixQubitOrder(originalMatrix, {1, 0});

  // Should be SWAP * original * SWAP
  auto expected = SWAP_GATE * originalMatrix * SWAP_GATE;
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, GetSingleQubitMatrixRX) {
  Gate gate{.type = qc::RX, .parameter = {1.5}, .qubitId = {0}};
  auto result = getSingleQubitMatrix(gate);
  auto expected = rxMatrix(1.5);
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, GetSingleQubitMatrixRY) {
  Gate gate{.type = qc::RY, .parameter = {2.3}, .qubitId = {0}};
  auto result = getSingleQubitMatrix(gate);
  auto expected = ryMatrix(2.3);
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, GetSingleQubitMatrixRZ) {
  Gate gate{.type = qc::RZ, .parameter = {0.7}, .qubitId = {0}};
  auto result = getSingleQubitMatrix(gate);
  auto expected = rzMatrix(0.7);
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, GetSingleQubitMatrixIdentity) {
  Gate gate{.type = qc::I, .parameter = {}, .qubitId = {0}};
  auto result = getSingleQubitMatrix(gate);
  EXPECT_TRUE(result.isApprox(IDENTITY_GATE, 1e-12));
}

TEST(UnitaryMatricesTest, GetSingleQubitMatrixH) {
  Gate gate{.type = qc::H, .parameter = {}, .qubitId = {0}};
  auto result = getSingleQubitMatrix(gate);
  EXPECT_TRUE(result.isApprox(H_GATE, 1e-12));
}

TEST(UnitaryMatricesTest, GetTwoQubitMatrixRXX) {
  Gate gate{.type = qc::RXX, .parameter = {1.5}, .qubitId = {0, 1}};
  auto result = getTwoQubitMatrix(gate);
  auto expected = rxxMatrix(1.5);
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, GetTwoQubitMatrixRYY) {
  Gate gate{.type = qc::RYY, .parameter = {2.3}, .qubitId = {0, 1}};
  auto result = getTwoQubitMatrix(gate);
  auto expected = ryyMatrix(2.3);
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, GetTwoQubitMatrixRZZ) {
  Gate gate{.type = qc::RZZ, .parameter = {0.7}, .qubitId = {0, 1}};
  auto result = getTwoQubitMatrix(gate);
  auto expected = rzzMatrix(0.7);
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, GetTwoQubitMatrixCNOT01) {
  Gate gate{.type = qc::X, .parameter = {}, .qubitId = {0, 1}};
  auto result = getTwoQubitMatrix(gate);

  matrix4x4 expected{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, GetTwoQubitMatrixCNOT10) {
  Gate gate{.type = qc::X, .parameter = {}, .qubitId = {1, 0}};
  auto result = getTwoQubitMatrix(gate);

  matrix4x4 expected{{1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}};
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(UnitaryMatricesTest, AllRotationMatricesAreUnitary) {
  std::vector<fp> angles = {0.0, qc::PI_4, qc::PI_2, qc::PI, -qc::PI_2};

  for (auto angle : angles) {
    EXPECT_TRUE(isUnitaryMatrix(rxMatrix(angle)));
    EXPECT_TRUE(isUnitaryMatrix(ryMatrix(angle)));
    EXPECT_TRUE(isUnitaryMatrix(rzMatrix(angle)));
    EXPECT_TRUE(isUnitaryMatrix(rxxMatrix(angle)));
    EXPECT_TRUE(isUnitaryMatrix(ryyMatrix(angle)));
    EXPECT_TRUE(isUnitaryMatrix(rzzMatrix(angle)));
  }
}

TEST(UnitaryMatricesTest, NegativeAngles) {
  // Test that negative angles produce valid unitary matrices
  EXPECT_TRUE(isUnitaryMatrix(rxMatrix(-1.5)));
  EXPECT_TRUE(isUnitaryMatrix(ryMatrix(-2.3)));
  EXPECT_TRUE(isUnitaryMatrix(rzMatrix(-0.7)));
}
