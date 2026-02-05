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
#include "mlir/Passes/Decomposition/BasisDecomposer.h"
#include "mlir/Passes/Decomposition/EulerBasis.h"
#include "mlir/Passes/Decomposition/Gate.h"
#include "mlir/Passes/Decomposition/GateSequence.h"
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
#include <llvm/ADT/SmallVector.h>
#include <optional>
#include <tuple>

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

class BasisDecomposerTest
    : public testing::TestWithParam<
          std::tuple<Gate, llvm::SmallVector<EulerBasis>, matrix4x4>> {
public:
  void SetUp() override {
    basisGate = std::get<0>(GetParam());
    eulerBases = std::get<1>(GetParam());
    target = std::get<2>(GetParam());
    targetDecomposition = TwoQubitWeylDecomposition::create(target, 1.0);
  }

  [[nodiscard]] static matrix4x4 restore(const TwoQubitGateSequence& sequence) {
    matrix4x4 matrix = matrix4x4::Identity();
    for (auto&& gate : sequence.gates) {
      matrix = getTwoQubitMatrix(gate) * matrix;
    }

    matrix *= std::exp(IM * sequence.globalPhase);
    return matrix;
  }

protected:
  matrix4x4 target;
  Gate basisGate;
  llvm::SmallVector<EulerBasis> eulerBases;
  TwoQubitWeylDecomposition targetDecomposition;
};

TEST_P(BasisDecomposerTest, TestExact) {
  const auto& originalMatrix = target;
  auto decomposer = TwoQubitBasisDecomposer::create(basisGate, 1.0);
  auto decomposedSequence = decomposer.twoQubitDecompose(
      targetDecomposition, eulerBases, 1.0, false, std::nullopt);

  ASSERT_TRUE(decomposedSequence.has_value());

  auto restoredMatrix = restore(*decomposedSequence);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix))
      << "RESULT:\n"
      << restoredMatrix << '\n';
}

TEST_P(BasisDecomposerTest, TestApproximation) {
  const auto& originalMatrix = target;
  auto decomposer = TwoQubitBasisDecomposer::create(basisGate, 1.0 - 1e-12);
  auto decomposedSequence = decomposer.twoQubitDecompose(
      targetDecomposition, eulerBases, 1.0 - 1e-12, true, std::nullopt);

  ASSERT_TRUE(decomposedSequence.has_value());

  auto restoredMatrix = restore(*decomposedSequence);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix))
      << "RESULT:\n"
      << restoredMatrix << '\n';
}

TEST(BasisDecomposerTest, Random) {
  auto stopTime = std::chrono::steady_clock::now() + std::chrono::seconds{10};
  auto iterations = 0;

  const Gate basisGate{.type = qc::X, .parameter = {}, .qubitId = {0, 1}};
  const llvm::SmallVector<EulerBasis> eulerBases = {EulerBasis::XYX,
                                                    EulerBasis::ZXZ};

  while (std::chrono::steady_clock::now() < stopTime) {
    auto originalMatrix = randomUnitaryMatrix();

    auto targetDecomposition =
        TwoQubitWeylDecomposition::create(originalMatrix, 1.0);
    auto decomposer = TwoQubitBasisDecomposer::create(basisGate, 1.0);
    auto decomposedSequence = decomposer.twoQubitDecompose(
        targetDecomposition, eulerBases, 1.0, true, std::nullopt);

    ASSERT_TRUE(decomposedSequence.has_value());

    auto restoredMatrix = BasisDecomposerTest::restore(*decomposedSequence);

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
    SingleQubitMatrices, BasisDecomposerTest,
    testing::Combine(
        // basis gates
        testing::Values(Gate{.type = qc::X, .parameter = {}, .qubitId = {0, 1}},
                        Gate{
                            .type = qc::X, .parameter = {}, .qubitId = {1, 0}}),
        // sets of euler bases
        testing::Values(llvm::SmallVector<EulerBasis>{EulerBasis::ZYZ},
                        llvm::SmallVector<EulerBasis>{
                            EulerBasis::ZYZ, EulerBasis::ZXZ, EulerBasis::XYX,
                            EulerBasis::XZX},
                        llvm::SmallVector<EulerBasis>{EulerBasis::XZX}),
        // targets to be decomposed
        testing::Values(helpers::kroneckerProduct(IDENTITY_GATE, IDENTITY_GATE),
                        helpers::kroneckerProduct(rzMatrix(1.0), ryMatrix(3.1)),
                        helpers::kroneckerProduct(IDENTITY_GATE,
                                                  rxMatrix(0.1)))));

INSTANTIATE_TEST_CASE_P(
    TwoQubitMatrices, BasisDecomposerTest,
    testing::Combine(
        // basis gates
        testing::Values(Gate{.type = qc::X, .parameter = {}, .qubitId = {0, 1}},
                        Gate{
                            .type = qc::X, .parameter = {}, .qubitId = {1, 0}}),
        // sets of euler bases
        testing::Values(llvm::SmallVector<EulerBasis>{EulerBasis::ZYZ},
                        llvm::SmallVector<EulerBasis>{
                            EulerBasis::ZYZ, EulerBasis::ZXZ, EulerBasis::XYX,
                            EulerBasis::XZX},
                        llvm::SmallVector<EulerBasis>{EulerBasis::XZX}),
        // targets to be decomposed
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
                helpers::kroneckerProduct(IPX, IPY))));

// Additional edge case tests for BasisDecomposer
TEST(BasisDecomposerEdgeCasesTest, ZeroAngleRotations) {
  const Gate basisGate{.type = qc::X, .parameter = {}, .qubitId = {0, 1}};
  const llvm::SmallVector<EulerBasis> eulerBases = {EulerBasis::ZYZ};

  // Test with zero angle rotations (should be close to identity)
  auto target = rxxMatrix(0.0) * ryyMatrix(0.0) * rzzMatrix(0.0);
  auto targetDecomposition = TwoQubitWeylDecomposition::create(target, 1.0);
  auto decomposer = TwoQubitBasisDecomposer::create(basisGate, 1.0);
  auto decomposedSequence = decomposer.twoQubitDecompose(
      targetDecomposition, eulerBases, 1.0, false, std::nullopt);

  ASSERT_TRUE(decomposedSequence.has_value());
  auto restoredMatrix = BasisDecomposerTest::restore(*decomposedSequence);
  EXPECT_TRUE(restoredMatrix.isApprox(target, 1e-12));
}

TEST(BasisDecomposerEdgeCasesTest, MaximalEntanglement) {
  const Gate basisGate{.type = qc::X, .parameter = {}, .qubitId = {0, 1}};
  const llvm::SmallVector<EulerBasis> eulerBases = {EulerBasis::XYX,
                                                    EulerBasis::ZXZ};

  // Test with maximally entangling gate (canonical gate at pi/4, pi/4, pi/4)
  auto target = canonicalGate(qc::PI_4, qc::PI_4, qc::PI_4);
  auto targetDecomposition = TwoQubitWeylDecomposition::create(target, 1.0);
  auto decomposer = TwoQubitBasisDecomposer::create(basisGate, 1.0);
  auto decomposedSequence = decomposer.twoQubitDecompose(
      targetDecomposition, eulerBases, 1.0, false, std::nullopt);

  ASSERT_TRUE(decomposedSequence.has_value());
  auto restoredMatrix = BasisDecomposerTest::restore(*decomposedSequence);
  EXPECT_TRUE(restoredMatrix.isApprox(target, 1e-12));
}

TEST(BasisDecomposerEdgeCasesTest, NegativeAngles) {
  const Gate basisGate{.type = qc::X, .parameter = {}, .qubitId = {0, 1}};
  const llvm::SmallVector<EulerBasis> eulerBases = {EulerBasis::ZYZ};

  // Test with negative angles
  auto target = rxxMatrix(-1.5) * ryyMatrix(-0.7) * rzzMatrix(-2.3);
  auto targetDecomposition = TwoQubitWeylDecomposition::create(target, 1.0);
  auto decomposer = TwoQubitBasisDecomposer::create(basisGate, 1.0);
  auto decomposedSequence = decomposer.twoQubitDecompose(
      targetDecomposition, eulerBases, 1.0, false, std::nullopt);

  ASSERT_TRUE(decomposedSequence.has_value());
  auto restoredMatrix = BasisDecomposerTest::restore(*decomposedSequence);
  EXPECT_TRUE(restoredMatrix.isApprox(target, 1e-12));
}

TEST(BasisDecomposerEdgeCasesTest, VerySmallAngles) {
  const Gate basisGate{.type = qc::X, .parameter = {}, .qubitId = {0, 1}};
  const llvm::SmallVector<EulerBasis> eulerBases = {EulerBasis::ZXZ};

  // Test with very small angles (near numerical precision)
  auto target = rxxMatrix(1e-10) * ryyMatrix(1e-11) * rzzMatrix(1e-12);
  auto targetDecomposition = TwoQubitWeylDecomposition::create(target, 1.0);
  auto decomposer = TwoQubitBasisDecomposer::create(basisGate, 1.0);
  auto decomposedSequence = decomposer.twoQubitDecompose(
      targetDecomposition, eulerBases, 1.0, false, std::nullopt);

  ASSERT_TRUE(decomposedSequence.has_value());
  auto restoredMatrix = BasisDecomposerTest::restore(*decomposedSequence);
  EXPECT_TRUE(restoredMatrix.isApprox(target, 1e-12));
}

TEST(BasisDecomposerEdgeCasesTest, AnglesAtPiBoundary) {
  const Gate basisGate{.type = qc::X, .parameter = {}, .qubitId = {0, 1}};
  const llvm::SmallVector<EulerBasis> eulerBases = {EulerBasis::XYX};

  // Test with angles at pi boundary
  auto target = rxxMatrix(qc::PI) * ryyMatrix(qc::PI / 2.0) * rzzMatrix(qc::PI);
  auto targetDecomposition = TwoQubitWeylDecomposition::create(target, 1.0);
  auto decomposer = TwoQubitBasisDecomposer::create(basisGate, 1.0);
  auto decomposedSequence = decomposer.twoQubitDecompose(
      targetDecomposition, eulerBases, 1.0, false, std::nullopt);

  ASSERT_TRUE(decomposedSequence.has_value());
  auto restoredMatrix = BasisDecomposerTest::restore(*decomposedSequence);
  EXPECT_TRUE(restoredMatrix.isApprox(target, 1e-12));
}

TEST(BasisDecomposerEdgeCasesTest, SwapGate) {
  const Gate basisGate{.type = qc::X, .parameter = {}, .qubitId = {0, 1}};
  const llvm::SmallVector<EulerBasis> eulerBases = {EulerBasis::ZYZ,
                                                    EulerBasis::XZX};

  // Test SWAP gate decomposition
  auto target = SWAP_GATE;
  auto targetDecomposition = TwoQubitWeylDecomposition::create(target, 1.0);
  auto decomposer = TwoQubitBasisDecomposer::create(basisGate, 1.0);
  auto decomposedSequence = decomposer.twoQubitDecompose(
      targetDecomposition, eulerBases, 1.0, false, std::nullopt);

  ASSERT_TRUE(decomposedSequence.has_value());
  auto restoredMatrix = BasisDecomposerTest::restore(*decomposedSequence);
  EXPECT_TRUE(restoredMatrix.isApprox(target, 1e-12));
}

TEST(BasisDecomposerEdgeCasesTest, ControlledGateWithPhase) {
  const Gate basisGate{.type = qc::X, .parameter = {}, .qubitId = {0, 1}};
  const llvm::SmallVector<EulerBasis> eulerBases = {EulerBasis::ZYZ};

  // Test controlled gate with additional phase gates
  auto target = getTwoQubitMatrix({.type = qc::X, .parameter = {}, .qubitId = {0, 1}}) *
                helpers::kroneckerProduct(pMatrix(qc::PI / 4.0), pMatrix(qc::PI / 3.0));
  auto targetDecomposition = TwoQubitWeylDecomposition::create(target, 1.0);
  auto decomposer = TwoQubitBasisDecomposer::create(basisGate, 1.0);
  auto decomposedSequence = decomposer.twoQubitDecompose(
      targetDecomposition, eulerBases, 1.0, false, std::nullopt);

  ASSERT_TRUE(decomposedSequence.has_value());
  auto restoredMatrix = BasisDecomposerTest::restore(*decomposedSequence);
  EXPECT_TRUE(restoredMatrix.isApprox(target, 1e-12));
}

TEST(BasisDecomposerEdgeCasesTest, ReversedQubitOrder) {
  const Gate basisGate{.type = qc::X, .parameter = {}, .qubitId = {1, 0}};
  const llvm::SmallVector<EulerBasis> eulerBases = {EulerBasis::XYX};

  // Test with reversed qubit order in basis gate
  auto target = canonicalGate(0.5, 0.3, 0.2);
  auto targetDecomposition = TwoQubitWeylDecomposition::create(target, 1.0);
  auto decomposer = TwoQubitBasisDecomposer::create(basisGate, 1.0);
  auto decomposedSequence = decomposer.twoQubitDecompose(
      targetDecomposition, eulerBases, 1.0, false, std::nullopt);

  ASSERT_TRUE(decomposedSequence.has_value());
  auto restoredMatrix = BasisDecomposerTest::restore(*decomposedSequence);
  EXPECT_TRUE(restoredMatrix.isApprox(target, 1e-12));
}

TEST(BasisDecomposerEdgeCasesTest, ComplexProductOfRotations) {
  const Gate basisGate{.type = qc::X, .parameter = {}, .qubitId = {0, 1}};
  const llvm::SmallVector<EulerBasis> eulerBases = {EulerBasis::ZXZ, EulerBasis::XYX};

  // Test complex product of rotations
  auto target = rzzMatrix(0.7) * ryyMatrix(1.2) * rxxMatrix(0.5) *
                helpers::kroneckerProduct(rzMatrix(0.3), ryMatrix(0.8)) *
                rzzMatrix(1.1) * ryyMatrix(0.9) * rxxMatrix(1.4);
  auto targetDecomposition = TwoQubitWeylDecomposition::create(target, 1.0);
  auto decomposer = TwoQubitBasisDecomposer::create(basisGate, 1.0);
  auto decomposedSequence = decomposer.twoQubitDecompose(
      targetDecomposition, eulerBases, 1.0, false, std::nullopt);

  ASSERT_TRUE(decomposedSequence.has_value());
  auto restoredMatrix = BasisDecomposerTest::restore(*decomposedSequence);
  EXPECT_TRUE(restoredMatrix.isApprox(target, 1e-12));
}