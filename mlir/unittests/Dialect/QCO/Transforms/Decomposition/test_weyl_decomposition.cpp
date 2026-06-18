/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "decomposition_test_utils.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Helpers.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/UnitaryMatrices.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/WeylDecomposition.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <gtest/gtest.h>

#include <complex>
#include <optional>
#include <random>

using namespace mlir::qco;
using namespace mlir::qco::decomposition;
using namespace mlir::qco::decomposition_test;

// NOLINTNEXTLINE(misc-use-internal-linkage) -- gtest `TEST_P` at global scope
class WeylDecompositionTest : public testing::TestWithParam<Matrix4x4 (*)()> {
public:
  [[nodiscard]] static Matrix4x4
  restore(const TwoQubitWeylDecomposition& decomposition) {
    return k1(decomposition) * can(decomposition) * k2(decomposition) *
           globalPhaseFactor(decomposition);
  }

  [[nodiscard]] static std::complex<double>
  globalPhaseFactor(const TwoQubitWeylDecomposition& decomposition) {
    return helpers::globalPhaseFactor(decomposition.globalPhase());
  }
  [[nodiscard]] static Matrix4x4
  can(const TwoQubitWeylDecomposition& decomposition) {
    return decomposition.getCanonicalMatrix();
  }
  [[nodiscard]] static Matrix4x4
  k1(const TwoQubitWeylDecomposition& decomposition) {
    return kron(decomposition.k1l(), decomposition.k1r());
  }
  [[nodiscard]] static Matrix4x4
  k2(const TwoQubitWeylDecomposition& decomposition) {
    return kron(decomposition.k2l(), decomposition.k2r());
  }
};

TEST_P(WeylDecompositionTest, TestExact) {
  const auto& originalMatrix = GetParam()();
  auto decomposition = TwoQubitWeylDecomposition::create(
      originalMatrix, std::optional<double>{1.0});
  auto restoredMatrix = restore(decomposition);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix));
}

TEST_P(WeylDecompositionTest, TestApproximation) {
  const auto& originalMatrix = GetParam()();
  auto decomposition = TwoQubitWeylDecomposition::create(
      originalMatrix, std::optional<double>{1.0 - 1e-12});
  auto restoredMatrix = restore(decomposition);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix));
}

TEST(WeylDecompositionStandalone,
     CnotProducesValidWeylParametersAndUnitaryLocals) {
  const Matrix4x4 cnot = Matrix4x4::fromElements(1, 0, 0, 0, // row 0
                                                 0, 1, 0, 0, // row 1
                                                 0, 0, 0, 1, // row 2
                                                 0, 0, 1, 0);

  const auto decomp = TwoQubitWeylDecomposition::create(cnot, std::nullopt);
  EXPECT_GE(decomp.a(), -1e-10);
  EXPECT_GE(decomp.b(), -1e-10);
  EXPECT_GE(decomp.c(), -1e-10);
  constexpr double piOver4 = 0.7853981633974483;
  EXPECT_LE(decomp.a(), piOver4 + 1e-10);
  EXPECT_LE(decomp.b(), piOver4 + 1e-10);
  EXPECT_LE(decomp.c(), piOver4 + 1e-10);
  EXPECT_TRUE(helpers::isUnitaryMatrix(decomp.k1l()));
  EXPECT_TRUE(helpers::isUnitaryMatrix(decomp.k2l()));
  EXPECT_TRUE(helpers::isUnitaryMatrix(decomp.k1r()));
  EXPECT_TRUE(helpers::isUnitaryMatrix(decomp.k2r()));
}

TEST(WeylDecompositionStandalone, Random) {
  constexpr auto maxIterations = 5000;
  std::mt19937 rng{1234567UL};

  for (int i = 0; i < maxIterations; ++i) {
    auto originalMatrix = randomUnitary4x4(rng);
    auto decomposition = TwoQubitWeylDecomposition::create(
        originalMatrix, std::optional<double>{1.0 - 1e-12});
    auto restoredMatrix = WeylDecompositionTest::restore(decomposition);

    // The reconstruction accuracy is bounded by the iterative diagonalization
    // residual rather than the (much tighter) default matrix tolerance.
    EXPECT_TRUE(
        restoredMatrix.isApprox(originalMatrix, SANITY_CHECK_PRECISION));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ProductTwoQubitMatrices, WeylDecompositionTest,
    ::testing::Values([]() -> Matrix4x4 { return Matrix4x4::identity(); },
                      []() -> Matrix4x4 {
                        return kron(rzMatrix(1.0), ryMatrix(3.1));
                      },
                      []() -> Matrix4x4 {
                        return kron(Matrix2x2::identity(), rxMatrix(0.1));
                      }));

INSTANTIATE_TEST_SUITE_P(
    TwoQubitMatrices, WeylDecompositionTest,
    ::testing::Values(
        []() -> Matrix4x4 { return rzzMatrix(2.0); },
        []() -> Matrix4x4 {
          return ryyMatrix(1.0) * rzzMatrix(3.0) * rxxMatrix(2.0);
        },
        []() -> Matrix4x4 {
          return TwoQubitWeylDecomposition::getCanonicalMatrix(1.5, -0.2, 0.0) *
                 kron(rxMatrix(1.0), Matrix2x2::identity());
        },
        []() -> Matrix4x4 {
          return kron(rxMatrix(1.0), ryMatrix(1.0)) *
                 TwoQubitWeylDecomposition::getCanonicalMatrix(1.1, 0.2, 3.0) *
                 kron(rxMatrix(1.0), Matrix2x2::identity());
        },
        []() -> Matrix4x4 {
          return kron(hGate(), ipz()) * cxGate01() * kron(ipx(), ipy());
        }));

INSTANTIATE_TEST_SUITE_P(
    SpecializedMatrices, WeylDecompositionTest,
    ::testing::Values(
        // id + controlled + general already covered by other parametrizations
        // swap equiv
        []() -> Matrix4x4 { return cxGate01() * cxGate10() * cxGate01(); },
        // partial swap equiv
        []() -> Matrix4x4 {
          return TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.5, 0.5);
        },
        // partial swap equiv (flipped)
        []() -> Matrix4x4 {
          return TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.5, -0.5);
        },
        // mirror controlled equiv
        []() -> Matrix4x4 { return cxGate01() * cxGate10(); },
        // sim aab equiv
        []() -> Matrix4x4 {
          return TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.5, 0.1);
        },
        // sim abb equiv
        []() -> Matrix4x4 {
          return TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.1, 0.1);
        },
        // sim ab-b equiv
        []() -> Matrix4x4 {
          return TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.1, -0.1);
        }));
