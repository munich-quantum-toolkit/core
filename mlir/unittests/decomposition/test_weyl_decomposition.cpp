/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/OpType.hpp"
#include "mlir/Passes/Decomposition/Helpers.h"
#include "mlir/Passes/Decomposition/UnitaryMatrices.h"
#include "mlir/Passes/Decomposition/WeylDecomposition.h"

#include <Eigen/QR>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>

using namespace mlir::qco;
using namespace mlir::qco::decomposition;

namespace {
[[nodiscard]] matrix4x4 randomUnitaryMatrix() {
  [[maybe_unused]] static auto initializeRandom = []() {
    // Eigen uses std::rand() internally, use fixed seed for deterministic
    // testing behavior
    std::srand(123456UL);
    return true;
  }();
  const matrix4x4 randomMatrix = matrix4x4::Random();
  Eigen::HouseholderQR<matrix4x4> qr{}; // NOLINT(misc-include-cleaner)
  qr.compute(randomMatrix);
  const matrix4x4 unitaryMatrix = qr.householderQ();
  assert(helpers::isUnitaryMatrix(unitaryMatrix));
  return unitaryMatrix;
}

[[nodiscard]] matrix4x4 canonicalGate(fp a, fp b, fp c) {
  TwoQubitWeylDecomposition tmp{};
  tmp.a = a;
  tmp.b = b;
  tmp.c = c;
  return tmp.getCanonicalMatrix();
}
} // namespace

class WeylDecompositionTest : public testing::TestWithParam<matrix4x4> {
public:
  [[nodiscard]] static matrix4x4
  restore(const TwoQubitWeylDecomposition& decomposition) {
    return k1(decomposition) * can(decomposition) * k2(decomposition) *
           globalPhaseFactor(decomposition);
  }

  [[nodiscard]] static qfp
  globalPhaseFactor(const TwoQubitWeylDecomposition& decomposition) {
    return std::exp(IM * decomposition.globalPhase);
  }
  [[nodiscard]] static matrix4x4
  can(const TwoQubitWeylDecomposition& decomposition) {
    return decomposition.getCanonicalMatrix();
  }
  [[nodiscard]] static matrix4x4
  k1(const TwoQubitWeylDecomposition& decomposition) {
    return helpers::kroneckerProduct(decomposition.k1l, decomposition.k1r);
  }
  [[nodiscard]] static matrix4x4
  k2(const TwoQubitWeylDecomposition& decomposition) {
    return helpers::kroneckerProduct(decomposition.k2l, decomposition.k2r);
  }
};

TEST_P(WeylDecompositionTest, TestExact) {
  const auto& originalMatrix = GetParam();
  auto decomposition = TwoQubitWeylDecomposition::create(originalMatrix, 1.0);
  auto restoredMatrix = restore(decomposition);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix))
      << "RESULT:\n"
      << restoredMatrix << '\n';
}

TEST_P(WeylDecompositionTest, TestApproximation) {
  const auto& originalMatrix = GetParam();
  auto decomposition =
      TwoQubitWeylDecomposition::create(originalMatrix, 1.0 - 1e-12);
  auto restoredMatrix = restore(decomposition);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix))
      << "RESULT:\n"
      << restoredMatrix << '\n';
}

TEST(WeylDecompositionTest, Random) {
  auto stopTime = std::chrono::steady_clock::now() + std::chrono::seconds{10};
  auto iterations = 0;
  while (std::chrono::steady_clock::now() < stopTime) {
    auto originalMatrix = randomUnitaryMatrix();
    auto decomposition =
        TwoQubitWeylDecomposition::create(originalMatrix, 1.0 - 1e-12);
    auto restoredMatrix = WeylDecompositionTest::restore(decomposition);

    EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix))
        << "ORIGINAL:\n"
        << originalMatrix << '\n'
        << "RESULT:\n"
        << restoredMatrix << '\n';
    ++iterations;
  }

  RecordProperty("iterations", iterations);
  std::cerr << "Iterations: " << iterations << '\n';
}

INSTANTIATE_TEST_CASE_P(
    SingleQubitMatrices, WeylDecompositionTest,
    ::testing::Values(helpers::kroneckerProduct(IDENTITY_GATE, IDENTITY_GATE),
                      helpers::kroneckerProduct(rzMatrix(1.0), ryMatrix(3.1)),
                      helpers::kroneckerProduct(IDENTITY_GATE, rxMatrix(0.1))));

INSTANTIATE_TEST_CASE_P(
    TwoQubitMatrices, WeylDecompositionTest,
    ::testing::Values(
        rzzMatrix(2.0), ryyMatrix(1.0) * rzzMatrix(3.0) * rxxMatrix(2.0),
        canonicalGate(1.5, -0.2, 0.0) *
            helpers::kroneckerProduct(rxMatrix(1.0), IDENTITY_GATE),
        helpers::kroneckerProduct(rxMatrix(1.0), ryMatrix(1.0)) *
            canonicalGate(1.1, 0.2, 3.0) *
            helpers::kroneckerProduct(rxMatrix(1.0), IDENTITY_GATE),
        helpers::kroneckerProduct(H_GATE, IPZ) *
            getTwoQubitMatrix(
                {.type = qc::X, .parameter = {}, .qubitId = {0, 1}}) *
            helpers::kroneckerProduct(IPX, IPY)));

INSTANTIATE_TEST_CASE_P(
    SpecializedMatrices, WeylDecompositionTest,
    ::testing::Values(
        // id + controlled + general already covered by other parametrizations
        // swap equiv
        getTwoQubitMatrix({.type = qc::X, .parameter = {}, .qubitId = {0, 1}}) *
            getTwoQubitMatrix(
                {.type = qc::X, .parameter = {}, .qubitId = {1, 0}}) *
            getTwoQubitMatrix(
                {.type = qc::X, .parameter = {}, .qubitId = {0, 1}}),
        // partial swap equiv
        canonicalGate(0.5, 0.5, 0.5),
        // partial swap equiv (flipped)
        canonicalGate(0.5, 0.5, -0.5),
        // mirror controlled equiv
        getTwoQubitMatrix({.type = qc::X, .parameter = {}, .qubitId = {0, 1}}) *
            getTwoQubitMatrix(
                {.type = qc::X, .parameter = {}, .qubitId = {1, 0}}),
        // sim aab equiv
        canonicalGate(0.5, 0.5, 0.1),
        // sim abb equiv
        canonicalGate(0.5, 0.1, 0.1),
        // sim ab-b equiv
        canonicalGate(0.5, 0.1, -0.1)));

// Additional edge case tests for WeylDecomposition
TEST(WeylDecompositionEdgeCasesTest, IdentityMatrix) {
  auto originalMatrix = helpers::kroneckerProduct(IDENTITY_GATE, IDENTITY_GATE);
  auto decomposition = TwoQubitWeylDecomposition::create(originalMatrix, 1.0);
  auto restoredMatrix = WeylDecompositionTest::restore(decomposition);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix, 1e-12));

  // Identity should have specialization IdEquiv
  EXPECT_EQ(decomposition.specialization,
            TwoQubitWeylDecomposition::Specialization::IdEquiv);
}

TEST(WeylDecompositionEdgeCasesTest, CNOTGate) {
  // CNOT gate (controlled-X)
  matrix4x4 originalMatrix{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
  auto decomposition = TwoQubitWeylDecomposition::create(originalMatrix, 1.0);
  auto restoredMatrix = WeylDecompositionTest::restore(decomposition);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix, 1e-12));

  // CNOT should have specialization ControlledEquiv
  EXPECT_EQ(decomposition.specialization,
            TwoQubitWeylDecomposition::Specialization::ControlledEquiv);
}

TEST(WeylDecompositionEdgeCasesTest, ZeroCanonicalParameters) {
  // All canonical parameters are zero (should be identity)
  auto originalMatrix = canonicalGate(0.0, 0.0, 0.0);
  auto decomposition = TwoQubitWeylDecomposition::create(originalMatrix, 1.0);
  auto restoredMatrix = WeylDecompositionTest::restore(decomposition);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix, 1e-12));
  EXPECT_NEAR(decomposition.a, 0.0, 1e-12);
  EXPECT_NEAR(decomposition.b, 0.0, 1e-12);
  EXPECT_NEAR(decomposition.c, 0.0, 1e-12);
}

TEST(WeylDecompositionEdgeCasesTest, MaximalCanonicalParameters) {
  // Maximal canonical parameters (SWAP gate)
  auto originalMatrix = canonicalGate(qc::PI_4, qc::PI_4, qc::PI_4);
  auto decomposition = TwoQubitWeylDecomposition::create(originalMatrix, 1.0);
  auto restoredMatrix = WeylDecompositionTest::restore(decomposition);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix, 1e-12));
  EXPECT_EQ(decomposition.specialization,
            TwoQubitWeylDecomposition::Specialization::SWAPEquiv);
}

TEST(WeylDecompositionEdgeCasesTest, SingleParameterNonZero) {
  // Only one canonical parameter is non-zero
  auto matrices = {
    canonicalGate(0.5, 0.0, 0.0),
    canonicalGate(0.0, 0.5, 0.0),
    canonicalGate(0.0, 0.0, 0.5)
  };

  for (const auto& originalMatrix : matrices) {
    auto decomposition = TwoQubitWeylDecomposition::create(originalMatrix, 1.0);
    auto restoredMatrix = WeylDecompositionTest::restore(decomposition);
    EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix, 1e-12));
  }
}

TEST(WeylDecompositionEdgeCasesTest, NegativeCanonicalParameters) {
  // Negative canonical parameters
  auto originalMatrix = canonicalGate(-0.3, -0.2, -0.1);
  auto decomposition = TwoQubitWeylDecomposition::create(originalMatrix, 1.0);
  auto restoredMatrix = WeylDecompositionTest::restore(decomposition);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix, 1e-12));
}

TEST(WeylDecompositionEdgeCasesTest, GlobalPhaseVariations) {
  // Test matrices with different global phases
  auto baseMatrix = canonicalGate(0.3, 0.2, 0.1);

  for (auto phase : {0.0, qc::PI_4, qc::PI_2, qc::PI, -qc::PI_4}) {
    auto originalMatrix = std::exp(qfp(0, phase)) * baseMatrix;
    auto decomposition = TwoQubitWeylDecomposition::create(originalMatrix, 1.0);
    auto restoredMatrix = WeylDecompositionTest::restore(decomposition);

    EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix, 1e-12))
        << "Failed for global phase: " << phase;
  }
}

TEST(WeylDecompositionEdgeCasesTest, K1K2UnitaryCheck) {
  auto originalMatrix = randomUnitaryMatrix();
  auto decomposition = TwoQubitWeylDecomposition::create(originalMatrix, 1.0);

  // Verify that K1 and K2 are unitary
  auto k1 = WeylDecompositionTest::k1(decomposition);
  auto k2 = WeylDecompositionTest::k2(decomposition);

  EXPECT_TRUE(helpers::isUnitaryMatrix(k1, 1e-12));
  EXPECT_TRUE(helpers::isUnitaryMatrix(k2, 1e-12));
}