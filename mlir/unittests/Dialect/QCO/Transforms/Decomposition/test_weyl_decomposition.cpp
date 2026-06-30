/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <gtest/gtest.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <tuple>
#include <vector>

using namespace mlir::qco;
using namespace mlir::qco::decomposition;

static constexpr Matrix4x4 kTwoQubitControlledX01 =
    Matrix4x4::fromElements(1.0, 0.0, 0.0, 0.0, //
                            0.0, 1.0, 0.0, 0.0, //
                            0.0, 0.0, 0.0, 1.0, //
                            0.0, 0.0, 1.0, 0.0);

static constexpr Matrix4x4 kTwoQubitControlledX10 =
    Matrix4x4::fromElements(1.0, 0.0, 0.0, 0.0, //
                            0.0, 0.0, 0.0, 1.0, //
                            0.0, 0.0, 1.0, 0.0, //
                            0.0, 1.0, 0.0, 0.0);

template <typename MatrixT>
static bool isUnitaryMatrix(const MatrixT& matrix,
                            const double tolerance = MATRIX_TOLERANCE) {
  return (matrix.adjoint() * matrix).isIdentity(tolerance);
}

static Matrix4x4 randomUnitary4x4(std::mt19937& rng) {
  std::normal_distribution normalDist(0.0, 1.0);
  std::vector columns(4, std::vector(4, std::complex{0.0, 0.0}));
  for (auto& column : columns) {
    for (auto& entry : column) {
      entry = std::complex<double>(normalDist(rng), normalDist(rng));
    }
  }
  for (std::size_t j = 0; j < 4; ++j) {
    for (std::size_t k = 0; k < j; ++k) {
      std::complex<double> projection{0.0, 0.0};
      for (std::size_t i = 0; i < 4; ++i) {
        projection += std::conj(columns[k][i]) * columns[j][i];
      }
      for (std::size_t i = 0; i < 4; ++i) {
        columns[j][i] -= projection * columns[k][i];
      }
    }
    double norm = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
      norm += std::norm(columns[j][i]);
    }
    norm = std::sqrt(norm);
    for (std::size_t i = 0; i < 4; ++i) {
      columns[j][i] /= norm;
    }
  }
  const Matrix4x4 unitary = Matrix4x4::fromElements(
      columns[0][0], columns[1][0], columns[2][0], columns[3][0], columns[0][1],
      columns[1][1], columns[2][1], columns[3][1], columns[0][2], columns[1][2],
      columns[2][2], columns[3][2], columns[0][3], columns[1][3], columns[2][3],
      columns[3][3]);
  assert(isUnitaryMatrix(unitary, WEYL_TOLERANCE));
  return unitary;
}

static auto productMatrixCases() {
  return ::testing::Values([]() { return Matrix4x4::identity(); },
                           []() {
                             return Matrix4x4::kron(RZOp::unitaryMatrix(1.0),
                                                    RYOp::unitaryMatrix(3.1));
                           },
                           []() {
                             return Matrix4x4::kron(Matrix2x2::identity(),
                                                    RXOp::unitaryMatrix(0.1));
                           });
}

static auto entangledMatrixCases() {
  return ::testing::Values(
      []() { return RZZOp::unitaryMatrix(2.0); },
      []() {
        return RYYOp::unitaryMatrix(1.0) * RZZOp::unitaryMatrix(3.0) *
               RXXOp::unitaryMatrix(2.0);
      },
      []() {
        return TwoQubitWeylDecomposition::getCanonicalMatrix(1.5, -0.2, 0.0) *
               Matrix4x4::kron(RXOp::unitaryMatrix(1.0), Matrix2x2::identity());
      },
      []() {
        return Matrix4x4::kron(RXOp::unitaryMatrix(1.0),
                               RYOp::unitaryMatrix(1.0)) *
               TwoQubitWeylDecomposition::getCanonicalMatrix(1.1, 0.2, 3.0) *
               Matrix4x4::kron(RXOp::unitaryMatrix(1.0), Matrix2x2::identity());
      },
      []() {
        return Matrix4x4::kron(HOp::getUnitaryMatrix(),
                               Complex{0.0, 1.0} * ZOp::getUnitaryMatrix()) *
               kTwoQubitControlledX01 *
               Matrix4x4::kron(Complex{0.0, 1.0} * XOp::getUnitaryMatrix(),
                               Complex{0.0, 1.0} * YOp::getUnitaryMatrix());
      });
}

static auto cxBasisCases() {
  return ::testing::Values([]() { return kTwoQubitControlledX01; },
                           []() { return kTwoQubitControlledX10; });
}

static auto specializedMatrixCases() {
  return ::testing::Values(
      []() {
        return kTwoQubitControlledX01 * kTwoQubitControlledX10 *
               kTwoQubitControlledX01;
      },
      []() {
        return TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.5, 0.5);
      },
      []() {
        return TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.5, -0.5);
      },
      []() { return kTwoQubitControlledX01 * kTwoQubitControlledX10; },
      []() {
        return TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.5, 0.1);
      },
      []() {
        return TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.1, 0.1);
      },
      []() {
        return TwoQubitWeylDecomposition::getCanonicalMatrix(0.5, 0.1, -0.1);
      });
}

TEST(DecompositionHelpersTest, MatrixUtilitySanity) {
  EXPECT_NEAR(std::abs(std::polar(1.0, 1.25)), 1.0, 1e-14);
  EXPECT_FALSE(isUnitaryMatrix(Matrix2x2::fromElements(2.0, 0.0, 0.0, 2.0)));
  EXPECT_TRUE(isUnitaryMatrix(Matrix2x2::identity()));
}

TEST(DecompositionHelpersTest, GateMatrixFactoriesMatchCanonicalForm) {
  for (const double theta : {0.0, 0.25, 1.0, 2.5, -1.3}) {
    EXPECT_TRUE(RXXOp::unitaryMatrix(theta).isApprox(
        TwoQubitWeylDecomposition::getCanonicalMatrix(-theta / 2.0, 0.0, 0.0),
        WEYL_TOLERANCE));
    EXPECT_TRUE(RYYOp::unitaryMatrix(theta).isApprox(
        TwoQubitWeylDecomposition::getCanonicalMatrix(0.0, -theta / 2.0, 0.0),
        WEYL_TOLERANCE));
    EXPECT_TRUE(RZZOp::unitaryMatrix(theta).isApprox(
        TwoQubitWeylDecomposition::getCanonicalMatrix(0.0, 0.0, -theta / 2.0),
        WEYL_TOLERANCE));
  }
}

TEST(DecompositionHelpersTest, CanonicalMatrixMatchesGateProduct) {
  for (const auto [a, b, c] : {std::tuple{0.3, 0.2, 0.1},
                               {0.5, 0.5, 0.5},
                               {0.5, 0.1, -0.1},
                               {1.1, 0.2, 3.0},
                               {-0.2, 0.3, 0.4}}) {
    const auto fromGates = RZZOp::unitaryMatrix(-2.0 * c) *
                           RYYOp::unitaryMatrix(-2.0 * b) *
                           RXXOp::unitaryMatrix(-2.0 * a);
    EXPECT_TRUE(TwoQubitWeylDecomposition::getCanonicalMatrix(a, b, c).isApprox(
        fromGates, WEYL_TOLERANCE));
  }
}

namespace {

class WeylDecompositionTest : public testing::TestWithParam<Matrix4x4 (*)()> {};

class BasisDecomposerTest : public testing::TestWithParam<
                                std::tuple<Matrix4x4 (*)(), Matrix4x4 (*)()>> {
protected:
  void SetUp() override {
    basisMatrix = std::get<0>(GetParam())();
    target = std::get<1>(GetParam())();
    targetDecomposition = std::make_unique<TwoQubitWeylDecomposition>(
        TwoQubitWeylDecomposition::create(target, 1.0));
  }

  Matrix4x4 target;
  Matrix4x4 basisMatrix;
  std::unique_ptr<TwoQubitWeylDecomposition> targetDecomposition;
};

} // namespace

TEST_P(WeylDecompositionTest, ReconstructsWithinRequestedFidelity) {
  const Matrix4x4 originalMatrix = GetParam()();
  for (const double fidelity : {1.0, WEYL_DEFAULT_FIDELITY}) {
    const auto decomposition =
        TwoQubitWeylDecomposition::create(originalMatrix, fidelity);
    EXPECT_TRUE(
        decomposition.unitaryMatrix().isApprox(originalMatrix, WEYL_TOLERANCE));
  }
}

TEST(WeylDecompositionStandalone,
     CnotProducesValidWeylParametersAndUnitaryLocals) {
  const Matrix4x4 cnot =
      Matrix4x4::fromElements(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0);
  const auto decomp = TwoQubitWeylDecomposition::create(cnot, std::nullopt);
  constexpr double piOver4 = 0.7853981633974483;
  for (const double angle : {decomp.a(), decomp.b(), decomp.c()}) {
    EXPECT_GE(angle, -1e-10);
    EXPECT_LE(angle, piOver4 + 1e-10);
  }
  EXPECT_TRUE(isUnitaryMatrix(decomp.k1l()));
  EXPECT_TRUE(isUnitaryMatrix(decomp.k2l()));
  EXPECT_TRUE(isUnitaryMatrix(decomp.k1r()));
  EXPECT_TRUE(isUnitaryMatrix(decomp.k2r()));
}

TEST(WeylDecompositionStandalone, Random) {
  std::mt19937 rng{1234567UL};
  for (int i = 0; i < 5000; ++i) {
    const Matrix4x4 originalMatrix = randomUnitary4x4(rng);
    const auto decomposition = TwoQubitWeylDecomposition::create(
        originalMatrix, std::optional<double>{WEYL_DEFAULT_FIDELITY});
    EXPECT_TRUE(
        decomposition.unitaryMatrix().isApprox(originalMatrix, WEYL_TOLERANCE));
  }
}

INSTANTIATE_TEST_SUITE_P(ProductTwoQubitMatrices, WeylDecompositionTest,
                         productMatrixCases());
INSTANTIATE_TEST_SUITE_P(TwoQubitMatrices, WeylDecompositionTest,
                         entangledMatrixCases());
INSTANTIATE_TEST_SUITE_P(SpecializedMatrices, WeylDecompositionTest,
                         specializedMatrixCases());

TEST_P(BasisDecomposerTest, ReconstructsWithinRequestedFidelity) {
  for (const double fidelity : {1.0, WEYL_DEFAULT_FIDELITY}) {
    const auto decomposer =
        TwoQubitBasisDecomposer::create(basisMatrix, fidelity);
    const auto decomposed =
        decomposer.twoQubitDecompose(*targetDecomposition, std::nullopt);
    ASSERT_TRUE(decomposed.has_value());
    EXPECT_TRUE(unitaryMatrix(*decomposed, basisMatrix)
                    .isApprox(target, WEYL_TOLERANCE));
  }
}

TEST(BasisDecomposerTest, Random) {
  std::mt19937 rng{123456UL};
  const mlir::SmallVector<Matrix4x4, 2> basisMatrices{kTwoQubitControlledX01,
                                                      kTwoQubitControlledX10};
  std::uniform_int_distribution<std::size_t> distBasisGate{0, 1};

  for (int i = 0; i < 2000; ++i) {
    const Matrix4x4 originalMatrix = randomUnitary4x4(rng);
    const auto targetDecomposition = TwoQubitWeylDecomposition::create(
        originalMatrix, std::optional<double>{1.0});
    const Matrix4x4 basisMatrix = basisMatrices[distBasisGate(rng)];
    const auto decomposer = TwoQubitBasisDecomposer::create(basisMatrix, 1.0);
    const auto decomposed =
        decomposer.twoQubitDecompose(targetDecomposition, std::nullopt);
    ASSERT_TRUE(decomposed.has_value());
    EXPECT_TRUE(unitaryMatrix(*decomposed, basisMatrix)
                    .isApprox(originalMatrix, WEYL_TOLERANCE));
  }
}

TEST(BasisDecomposerNumBasisTest, ForcesZeroBasisUsesForIdentityTarget) {
  const Matrix4x4 basis = kTwoQubitControlledX01;
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const Matrix4x4 target = Matrix4x4::identity();
  const auto weyl = TwoQubitWeylDecomposition::create(target, 1.0);
  const auto decomposed = decomposer.twoQubitDecompose(weyl, std::uint8_t{0});
  ASSERT_TRUE(decomposed.has_value());
  EXPECT_EQ(decomposed->numBasisUses, 0);
  EXPECT_TRUE(
      unitaryMatrix(*decomposed, basis).isApprox(target, WEYL_TOLERANCE));
}

TEST(BasisDecomposerTest, DecomposeTwoQubitWithBasisReconstructsTarget) {
  const Matrix4x4 basis = kTwoQubitControlledX01;
  const Matrix4x4 target =
      Matrix4x4::kron(RXOp::unitaryMatrix(0.4), RYOp::unitaryMatrix(0.6)) *
      TwoQubitWeylDecomposition::getCanonicalMatrix(0.3, 0.2, 0.1) *
      Matrix4x4::kron(RZOp::unitaryMatrix(0.2), Matrix2x2::identity());
  const auto decomposed = decomposeTwoQubitWithBasis(target, basis);
  ASSERT_TRUE(decomposed.has_value());
  EXPECT_TRUE(
      unitaryMatrix(*decomposed, basis).isApprox(target, WEYL_TOLERANCE));
}

TEST(BasisDecomposerTest, CachedDecomposerMatchesOneShotAcrossTargets) {
  const Matrix4x4 basis = kTwoQubitControlledX01;
  const auto cachedDecomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const mlir::SmallVector<Matrix4x4, 3> targets{
      Matrix4x4::identity(),
      kTwoQubitControlledX01,
      Matrix4x4::kron(RXOp::unitaryMatrix(0.2), RYOp::unitaryMatrix(0.3)) *
          TwoQubitWeylDecomposition::getCanonicalMatrix(0.1, 0.2, 0.3) *
          Matrix4x4::kron(RZOp::unitaryMatrix(0.1), Matrix2x2::identity()),
  };
  for (const Matrix4x4& target : targets) {
    const auto oneShot = decomposeTwoQubitWithBasis(target, basis);
    const auto cached = cachedDecomposer.decomposeTarget(target);
    ASSERT_TRUE(oneShot.has_value());
    ASSERT_TRUE(cached.has_value());
    EXPECT_TRUE(
        unitaryMatrix(*oneShot, basis).isApprox(target, WEYL_TOLERANCE));
    EXPECT_TRUE(unitaryMatrix(*cached, basis).isApprox(target, WEYL_TOLERANCE));
    EXPECT_EQ(cached->numBasisUses, oneShot->numBasisUses);
    EXPECT_EQ(cached->singleQubitFactors.size(),
              oneShot->singleQubitFactors.size());
  }
}

TEST(BasisDecomposerTest, RejectsMultipleBasisUsesForNonSuperControlledBasis) {
  const Matrix4x4 basis = RZZOp::unitaryMatrix(1.0);
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const auto weyl =
      TwoQubitWeylDecomposition::create(Matrix4x4::identity(), 1.0);
  EXPECT_FALSE(decomposer.twoQubitDecompose(weyl, std::uint8_t{2}).has_value());
}

TEST(BasisDecomposerTest, RejectsInvalidBasisGateUseCount) {
  const Matrix4x4 basis = kTwoQubitControlledX01;
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const auto weyl =
      TwoQubitWeylDecomposition::create(kTwoQubitControlledX01, 1.0);
  EXPECT_FALSE(decomposer.twoQubitDecompose(weyl, std::uint8_t{4}).has_value());
}

TEST(BasisDecomposerForcedCountTest, OneBasisUseProducesFactors) {
  const Matrix4x4 basis = kTwoQubitControlledX01;
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const auto weyl =
      TwoQubitWeylDecomposition::create(kTwoQubitControlledX01, 1.0);
  const auto decomposed = decomposer.twoQubitDecompose(weyl, std::uint8_t{1});
  ASSERT_TRUE(decomposed.has_value());
  EXPECT_EQ(decomposed->numBasisUses, 1);
  EXPECT_EQ(decomposed->singleQubitFactors.size(), 4U);
}

TEST(BasisDecomposerForcedCountTest, TwoBasisUsesProducesFactors) {
  const Matrix4x4 basis = kTwoQubitControlledX01;
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const auto weyl =
      TwoQubitWeylDecomposition::create(kTwoQubitControlledX01, 1.0);
  const auto decomposed = decomposer.twoQubitDecompose(weyl, std::uint8_t{2});
  ASSERT_TRUE(decomposed.has_value());
  EXPECT_EQ(decomposed->numBasisUses, 2);
  EXPECT_EQ(decomposed->singleQubitFactors.size(), 6U);
}

TEST(BasisDecomposerForcedCountTest, ThreeBasisUsesProducesFactors) {
  const Matrix4x4 basis = kTwoQubitControlledX01;
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const auto weyl =
      TwoQubitWeylDecomposition::create(kTwoQubitControlledX01, 1.0);
  const auto decomposed = decomposer.twoQubitDecompose(weyl, std::uint8_t{3});
  ASSERT_TRUE(decomposed.has_value());
  EXPECT_EQ(decomposed->numBasisUses, 3);
  EXPECT_EQ(decomposed->singleQubitFactors.size(), 8U);
}

TEST(WeylDecompositionStandalone, SwapNegativeCSpecializationReconstructs) {
  constexpr double piOver4 = std::numbers::pi / 4.0;
  const Matrix4x4 swapNegativeC =
      TwoQubitWeylDecomposition::getCanonicalMatrix(piOver4, piOver4, -piOver4);
  const auto decomposition =
      TwoQubitWeylDecomposition::create(swapNegativeC, 1.0);
  EXPECT_TRUE(
      decomposition.unitaryMatrix().isApprox(swapNegativeC, WEYL_TOLERANCE));
}

TEST(WeylDecompositionStandalone, ControlledSpecializationReconstructs) {
  const Matrix4x4 controlledLike =
      Matrix4x4::kron(RXOp::unitaryMatrix(0.3), RYOp::unitaryMatrix(0.4)) *
      TwoQubitWeylDecomposition::getCanonicalMatrix(0.6, 0.0, 0.0) *
      Matrix4x4::kron(Matrix2x2::identity(), RZOp::unitaryMatrix(0.2));
  const auto decomposition =
      TwoQubitWeylDecomposition::create(controlledLike, 1.0);
  EXPECT_TRUE(
      decomposition.unitaryMatrix().isApprox(controlledLike, WEYL_TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(ProductTwoQubitMatrices, BasisDecomposerTest,
                         testing::Combine(cxBasisCases(),
                                          productMatrixCases()));
INSTANTIATE_TEST_SUITE_P(TwoQubitMatrices, BasisDecomposerTest,
                         testing::Combine(cxBasisCases(),
                                          entangledMatrixCases()));
