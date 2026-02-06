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
#include "utils.h"

#include <Eigen/QR>
#include <cassert>
#include <gtest/gtest.h>
#include <unsupported/Eigen/KroneckerProduct>

using namespace mlir::qco;
using namespace mlir::qco::decomposition;

class WeylDecompositionTest : public testing::TestWithParam<Eigen::Matrix4cd> {
public:
  [[nodiscard]] static Eigen::Matrix4cd
  restore(const TwoQubitWeylDecomposition& decomposition) {
    return k1(decomposition) * can(decomposition) * k2(decomposition) *
           globalPhaseFactor(decomposition);
  }

  [[nodiscard]] static std::complex<double>
  globalPhaseFactor(const TwoQubitWeylDecomposition& decomposition) {
    return helpers::globalPhaseFactor(decomposition.globalPhase());
  }
  [[nodiscard]] static Eigen::Matrix4cd
  can(const TwoQubitWeylDecomposition& decomposition) {
    return decomposition.getCanonicalMatrix();
  }
  [[nodiscard]] static Eigen::Matrix4cd
  k1(const TwoQubitWeylDecomposition& decomposition) {
    return Eigen::kroneckerProduct(decomposition.k1l(), decomposition.k1r());
  }
  [[nodiscard]] static Eigen::Matrix4cd
  k2(const TwoQubitWeylDecomposition& decomposition) {
    return Eigen::kroneckerProduct(decomposition.k2l(), decomposition.k2r());
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
  constexpr auto maxIterations = 5000;
  for (int i = 0; i < maxIterations; ++i) {
    auto originalMatrix = randomUnitaryMatrix<Eigen::Matrix4cd>();
    auto decomposition =
        TwoQubitWeylDecomposition::create(originalMatrix, 1.0 - 1e-12);
    auto restoredMatrix = WeylDecompositionTest::restore(decomposition);

    EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix))
        << "ORIGINAL:\n"
        << originalMatrix << '\n'
        << "RESULT:\n"
        << restoredMatrix << '\n';
  }
}

INSTANTIATE_TEST_SUITE_P(
    SingleQubitMatrices, WeylDecompositionTest,
    ::testing::Values(Eigen::Matrix4cd::Identity(),
                      Eigen::kroneckerProduct(rzMatrix(1.0), ryMatrix(3.1)),
                      Eigen::kroneckerProduct(Eigen::Matrix2cd::Identity(),
                                              rxMatrix(0.1))));

INSTANTIATE_TEST_SUITE_P(
    TwoQubitMatrices, WeylDecompositionTest,
    ::testing::Values(
        rzzMatrix(2.0), ryyMatrix(1.0) * rzzMatrix(3.0) * rxxMatrix(2.0),
        TwoQubitWeylDecomposition::getCanonicalMatrix(1.5, -0.2, 0.0) *
            Eigen::kroneckerProduct(rxMatrix(1.0),
                                    Eigen::Matrix2cd::Identity()),
        Eigen::kroneckerProduct(rxMatrix(1.0), ryMatrix(1.0)) *
            TwoQubitWeylDecomposition::getCanonicalMatrix(1.1, 0.2, 3.0) *
            Eigen::kroneckerProduct(rxMatrix(1.0),
                                    Eigen::Matrix2cd::Identity()),
        Eigen::kroneckerProduct(H_GATE, IPZ) *
            getTwoQubitMatrix(
                {.type = qc::X, .parameter = {}, .qubitId = {0, 1}}) *
            Eigen::kroneckerProduct(IPX, IPY)));

INSTANTIATE_TEST_SUITE_P(
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
        TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.5, 0.5),
        // partial swap equiv (flipped)
        TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.5, -0.5),
        // mirror controlled equiv
        getTwoQubitMatrix({.type = qc::X, .parameter = {}, .qubitId = {0, 1}}) *
            getTwoQubitMatrix(
                {.type = qc::X, .parameter = {}, .qubitId = {1, 0}}),
        // sim aab equiv
        TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.5, 0.1),
        // sim abb equiv
        TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.1, 0.1),
        // sim ab-b equiv
        TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.1, -0.1)));
