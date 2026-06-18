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
#include "mlir/Dialect/QCO/Transforms/Decomposition/BasisDecomposer.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Helpers.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/UnitaryMatrices.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/WeylDecomposition.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <tuple>

using namespace mlir::qco;
using namespace mlir::qco::decomposition;
using namespace mlir::qco::decomposition_test;

// NOLINTNEXTLINE(misc-use-internal-linkage) -- gtest `TEST_P` at global scope
class BasisDecomposerTest : public testing::TestWithParam<
                                std::tuple<Matrix4x4 (*)(), Matrix4x4 (*)()>> {
public:
  /// Reconstruct the 4x4 unitary realized by a native two-qubit decomposition.
  ///
  /// The factors come in `(r, l)` pairs: `factors[2i]` acts on qubit 1 (LSB)
  /// and `factors[2i + 1]` on qubit 0 (MSB), mirroring `emitTwoQubitNative`.
  /// Each interior pair is followed by one entangler, with a trailing pair
  /// after the last entangler.
  [[nodiscard]] static Matrix4x4
  restore(const TwoQubitNativeDecomposition& decomposition,
          const Matrix4x4& entangler) {
    const auto& factors = decomposition.singleQubitFactors;
    const auto layer = [&](std::size_t i) {
      return kron(factors[(2 * i) + 1], factors[2 * i]);
    };
    Matrix4x4 matrix = layer(0);
    for (std::uint8_t i = 0; i < decomposition.numBasisUses; ++i) {
      matrix = entangler * matrix;
      matrix = layer(static_cast<std::size_t>(i) + 1) * matrix;
    }
    return matrix * helpers::globalPhaseFactor(decomposition.globalPhase);
  }

protected:
  void SetUp() override {
    basisMatrix = std::get<0>(GetParam())();
    target = std::get<1>(GetParam())();
    targetDecomposition = std::make_unique<TwoQubitWeylDecomposition>(
        TwoQubitWeylDecomposition::create(target, std::optional<double>{1.0}));
  }

  Matrix4x4 target;
  Matrix4x4 basisMatrix;
  std::unique_ptr<TwoQubitWeylDecomposition> targetDecomposition;
};

TEST_P(BasisDecomposerTest, TestExact) {
  const auto& originalMatrix = target;
  auto decomposer = TwoQubitBasisDecomposer::create(basisMatrix, 1.0);
  auto decomposed =
      decomposer.twoQubitDecompose(*targetDecomposition, std::nullopt);

  ASSERT_TRUE(decomposed.has_value());

  auto restoredMatrix = restore(*decomposed, basisMatrix);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix));
}

TEST_P(BasisDecomposerTest, TestApproximation) {
  const auto& originalMatrix = target;
  auto decomposer = TwoQubitBasisDecomposer::create(basisMatrix, 1.0 - 1e-12);
  auto decomposed =
      decomposer.twoQubitDecompose(*targetDecomposition, std::nullopt);

  ASSERT_TRUE(decomposed.has_value());

  auto restoredMatrix = restore(*decomposed, basisMatrix);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix));
}

TEST(BasisDecomposerTest, Random) {
  constexpr auto maxIterations = 2000;
  std::mt19937 rng{123456UL};

  const llvm::SmallVector<Matrix4x4, 2> basisMatrices{cxGate01(), cxGate10()};
  std::uniform_int_distribution<std::size_t> distBasisGate{
      0, basisMatrices.size() - 1};
  auto selectRandomBasisMatrix = [&]() {
    return basisMatrices[distBasisGate(rng)];
  };

  for (int i = 0; i < maxIterations; ++i) {
    auto originalMatrix = randomUnitary4x4(rng);

    auto targetDecomposition = TwoQubitWeylDecomposition::create(
        originalMatrix, std::optional<double>{1.0});
    const auto basisMatrix = selectRandomBasisMatrix();
    auto decomposer = TwoQubitBasisDecomposer::create(basisMatrix, 1.0);
    auto decomposed =
        decomposer.twoQubitDecompose(targetDecomposition, std::nullopt);

    ASSERT_TRUE(decomposed.has_value());

    auto restoredMatrix =
        BasisDecomposerTest::restore(*decomposed, basisMatrix);

    // Reconstruction accumulates the Weyl diagonalization residual through up
    // to three entangler layers, so allow a correspondingly relaxed tolerance.
    EXPECT_TRUE(
        restoredMatrix.isApprox(originalMatrix, SANITY_CHECK_PRECISION));
  }
}

TEST(BasisDecomposerNumBasisTest, ForcesZeroBasisUsesForIdentityTarget) {
  const auto basis = cxGate01();
  const auto decomposer = TwoQubitBasisDecomposer::create(basis, 1.0);
  const Matrix4x4 target = Matrix4x4::identity();
  const auto weyl =
      TwoQubitWeylDecomposition::create(target, std::optional<double>{1.0});
  const auto decomposed = decomposer.twoQubitDecompose(weyl, std::uint8_t{0});
  ASSERT_TRUE(decomposed.has_value());
  EXPECT_EQ(decomposed->numBasisUses, 0);
  const Matrix4x4 restored = BasisDecomposerTest::restore(*decomposed, basis);
  EXPECT_TRUE(restored.isApprox(target));
}

INSTANTIATE_TEST_SUITE_P(
    ProductTwoQubitMatrices, BasisDecomposerTest,
    testing::Combine(
        // basis entanglers
        testing::Values([]() -> Matrix4x4 { return cxGate01(); },
                        []() -> Matrix4x4 { return cxGate10(); }),
        // targets to be decomposed
        testing::Values([]() -> Matrix4x4 { return Matrix4x4::identity(); },
                        []() -> Matrix4x4 {
                          return kron(rzMatrix(1.0), ryMatrix(3.1));
                        },
                        []() -> Matrix4x4 {
                          return kron(Matrix2x2::identity(), rxMatrix(0.1));
                        })));

INSTANTIATE_TEST_SUITE_P(
    TwoQubitMatrices, BasisDecomposerTest,
    testing::Combine(
        // basis entanglers
        testing::Values([]() -> Matrix4x4 { return cxGate01(); },
                        []() -> Matrix4x4 { return cxGate10(); }),
        // targets to be decomposed
        ::testing::Values(
            []() -> Matrix4x4 { return rzzMatrix(2.0); },
            []() -> Matrix4x4 {
              return ryyMatrix(1.0) * rzzMatrix(3.0) * rxxMatrix(2.0);
            },
            []() -> Matrix4x4 {
              return TwoQubitWeylDecomposition::getCanonicalMatrix(1.5, -0.2,
                                                                   0.0) *
                     kron(rxMatrix(1.0), Matrix2x2::identity());
            },
            []() -> Matrix4x4 {
              return kron(rxMatrix(1.0), ryMatrix(1.0)) *
                     TwoQubitWeylDecomposition::getCanonicalMatrix(1.1, 0.2,
                                                                   3.0) *
                     kron(rxMatrix(1.0), Matrix2x2::identity());
            },
            []() -> Matrix4x4 {
              return kron(hGate(), ipz()) * cxGate01() * kron(ipx(), ipy());
            })));
