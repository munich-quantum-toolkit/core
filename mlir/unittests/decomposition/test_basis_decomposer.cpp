/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/MQTOpt/Transforms/Decomposition/BasisDecomposer.h"
#include "mlir/Dialect/MQTOpt/Transforms/Decomposition/Helpers.h"
#include "mlir/Dialect/MQTOpt/Transforms/Decomposition/UnitaryMatrices.h"

#include <Eigen/QR>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>

using namespace mqt::ir::opt;
using namespace mqt::ir::opt::decomposition;

namespace {
[[nodiscard]] matrix4x4 randomUnitaryMatrix() {
  [[maybe_unused]] static auto initializeRandom = []() {
    // Eigen uses std::rand() internally, use fixed seed for deterministic
    // testing behavior
    std::srand(123456UL);
    return true;
  }();
  const matrix4x4 randomMatrix = matrix4x4::Random();
  Eigen::HouseholderQR<matrix4x4> qr{};
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
  while (std::chrono::steady_clock::now() < stopTime) {
    auto originalMatrix = randomUnitaryMatrix();

    auto targetDecomposition =
        TwoQubitWeylDecomposition::create(originalMatrix, 1.0);

    Gate basisGate{.type = qc::X, .qubitId = {0, 1}};
    llvm::SmallVector<EulerBasis> eulerBases = {EulerBasis::XYX,
                                                EulerBasis::ZXZ};
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
        testing::Values(Gate{.type = qc::X, .qubitId = {0, 1}},
                        Gate{.type = qc::X, .qubitId = {1, 0}}),
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
        testing::Values(Gate{.type = qc::X, .qubitId = {0, 1}},
                        Gate{.type = qc::X, .qubitId = {1, 0}}),
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
                getTwoQubitMatrix({.type = qc::X, .qubitId = {0, 1}}) *
                helpers::kroneckerProduct(IPX, IPY))));
