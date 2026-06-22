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
#include <optional>
#include <random>
#include <tuple>
#include <vector>

using namespace mlir::qco;
using namespace mlir::qco::decomposition;

static Complex globalPhaseFactor(const double phase) {
  return std::exp(Complex{0.0, 1.0} * phase);
}

static const Matrix4x4& twoQubitControlledX01() {
  static const Matrix4x4 MATRIX = Matrix4x4::fromElements(1.0, 0.0, 0.0, 0.0, //
                                                          0.0, 1.0, 0.0, 0.0, //
                                                          0.0, 0.0, 0.0, 1.0, //
                                                          0.0, 0.0, 1.0, 0.0);
  return MATRIX;
}

static const Matrix4x4& twoQubitControlledX10() {
  static const Matrix4x4 MATRIX = Matrix4x4::fromElements(1.0, 0.0, 0.0, 0.0, //
                                                          0.0, 0.0, 0.0, 1.0, //
                                                          0.0, 0.0, 1.0, 0.0, //
                                                          0.0, 1.0, 0.0, 0.0);
  return MATRIX;
}

static bool isUnitaryMatrix(const Matrix2x2& matrix,
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
  assert((unitary.adjoint() * unitary).isIdentity(WEYL_TOLERANCE));
  return unitary;
}

static Matrix4x4 restoreWeyl(const TwoQubitWeylDecomposition& decomposition) {
  return Matrix4x4::kron(decomposition.k1l(), decomposition.k1r()) *
         decomposition.getCanonicalMatrix() *
         Matrix4x4::kron(decomposition.k2l(), decomposition.k2r()) *
         globalPhaseFactor(decomposition.globalPhase());
}

static Matrix4x4 restoreBasis(const TwoQubitNativeDecomposition& decomposition,
                              const Matrix4x4& entangler) {
  const auto& factors = decomposition.singleQubitFactors;
  const auto layer = [&](std::size_t i) {
    return Matrix4x4::kron(factors[(2 * i) + 1], factors[2 * i]);
  };
  Matrix4x4 matrix = layer(0);
  for (std::uint8_t i = 0; i < decomposition.numBasisUses; ++i) {
    matrix = entangler * matrix;
    matrix = layer(static_cast<std::size_t>(i) + 1) * matrix;
  }
  return matrix * globalPhaseFactor(decomposition.globalPhase);
}

static auto productMatrixCases() {
  return ::testing::Values(
      []() { return Matrix4x4::identity(); },
      []() { return Matrix4x4::kron(rzMatrix(1.0), ryMatrix(3.1)); },
      []() { return Matrix4x4::kron(Matrix2x2::identity(), rxMatrix(0.1)); });
}

static auto entangledMatrixCases() {
  return ::testing::Values(
      []() { return rzzMatrix(2.0); },
      []() { return ryyMatrix(1.0) * rzzMatrix(3.0) * rxxMatrix(2.0); },
      []() {
        return TwoQubitWeylDecomposition::getCanonicalMatrix(1.5, -0.2, 0.0) *
               Matrix4x4::kron(rxMatrix(1.0), Matrix2x2::identity());
      },
      []() {
        return Matrix4x4::kron(rxMatrix(1.0), ryMatrix(1.0)) *
               TwoQubitWeylDecomposition::getCanonicalMatrix(1.1, 0.2, 3.0) *
               Matrix4x4::kron(rxMatrix(1.0), Matrix2x2::identity());
      },
      []() {
        return Matrix4x4::kron(HOp::getUnitaryMatrix(), iPauliZ()) *
               twoQubitControlledX01() * Matrix4x4::kron(iPauliX(), iPauliY());
      });
}

static auto cxBasisCases() {
  return ::testing::Values([]() { return twoQubitControlledX01(); },
                           []() { return twoQubitControlledX10(); });
}

TEST(DecompositionHelpersTest, MatrixUtilitySanity) {
  EXPECT_NEAR(std::abs(globalPhaseFactor(1.25)), 1.0, 1e-14);
  EXPECT_FALSE(isUnitaryMatrix(Matrix2x2::fromElements(2.0, 0.0, 0.0, 2.0)));
  EXPECT_TRUE(isUnitaryMatrix(Matrix2x2::identity()));
}

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

TEST_P(WeylDecompositionTest, ReconstructsWithinRequestedFidelity) {
  const Matrix4x4 originalMatrix = GetParam()();
  for (const double fidelity : {1.0, 1.0 - 1e-12}) {
    const auto decomposition =
        TwoQubitWeylDecomposition::create(originalMatrix, fidelity);
    EXPECT_TRUE(restoreWeyl(decomposition).isApprox(originalMatrix));
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
        originalMatrix, std::optional<double>{1.0 - 1e-12});
    EXPECT_TRUE(
        restoreWeyl(decomposition).isApprox(originalMatrix, WEYL_TOLERANCE));
  }
}

INSTANTIATE_TEST_SUITE_P(ProductTwoQubitMatrices, WeylDecompositionTest,
                         productMatrixCases());
INSTANTIATE_TEST_SUITE_P(TwoQubitMatrices, WeylDecompositionTest,
                         entangledMatrixCases());

TEST_P(BasisDecomposerTest, ReconstructsWithinRequestedFidelity) {
  for (const double fidelity : {1.0, 1.0 - 1e-12}) {
    const auto decomposer =
        TwoQubitBasisDecomposer::create(basisMatrix, fidelity);
    const auto decomposed =
        decomposer.twoQubitDecompose(*targetDecomposition, std::nullopt);
    ASSERT_TRUE(decomposed.has_value());
    EXPECT_TRUE(restoreBasis(*decomposed, basisMatrix).isApprox(target));
  }
}

TEST(BasisDecomposerTest, Random) {
  std::mt19937 rng{123456UL};
  const mlir::SmallVector<Matrix4x4, 2> basisMatrices{twoQubitControlledX01(),
                                                      twoQubitControlledX10()};
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
    EXPECT_TRUE(restoreBasis(*decomposed, basisMatrix)
                    .isApprox(originalMatrix, WEYL_TOLERANCE));
  }
}

TEST(BasisDecomposerNumBasisTest, ForcesZeroBasisUsesForIdentityTarget) {
  const Matrix4x4 basis = twoQubitControlledX01();
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const Matrix4x4 target = Matrix4x4::identity();
  const auto weyl = TwoQubitWeylDecomposition::create(target, 1.0);
  const auto decomposed = decomposer.twoQubitDecompose(weyl, std::uint8_t{0});
  ASSERT_TRUE(decomposed.has_value());
  EXPECT_EQ(decomposed->numBasisUses, 0);
  EXPECT_TRUE(restoreBasis(*decomposed, basis).isApprox(target));
}

INSTANTIATE_TEST_SUITE_P(ProductTwoQubitMatrices, BasisDecomposerTest,
                         testing::Combine(cxBasisCases(),
                                          productMatrixCases()));
INSTANTIATE_TEST_SUITE_P(TwoQubitMatrices, BasisDecomposerTest,
                         testing::Combine(cxBasisCases(),
                                          entangledMatrixCases()));
