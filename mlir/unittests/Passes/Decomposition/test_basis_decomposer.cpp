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
#include <unsupported/Eigen/KroneckerProduct>

using namespace mlir::qco;
using namespace mlir::qco::decomposition;

namespace {
[[nodiscard]] Eigen::Matrix4cd randomUnitaryMatrix() {
  [[maybe_unused]] static auto initializeRandom = []() {
    // Eigen uses std::rand() internally, use fixed seed for deterministic
    // testing behavior
    std::srand(123456UL);
    return true;
  }();
  const Eigen::Matrix4cd randomMatrix = Eigen::Matrix4cd::Random();
  Eigen::HouseholderQR<Eigen::Matrix4cd> qr{}; // NOLINT(misc-include-cleaner)
  qr.compute(randomMatrix);
  const Eigen::Matrix4cd unitaryMatrix = qr.householderQ();
  assert(helpers::isUnitaryMatrix(unitaryMatrix));
  return unitaryMatrix;
}

[[nodiscard]] Eigen::Matrix4cd canonicalGate(double a, double b, double c) {
  TwoQubitWeylDecomposition tmp{};
  tmp.a = a;
  tmp.b = b;
  tmp.c = c;
  return tmp.getCanonicalMatrix();
}
} // namespace

class BasisDecomposerTest
    : public testing::TestWithParam<
          std::tuple<Gate, llvm::SmallVector<EulerBasis>, Eigen::Matrix4cd>> {
public:
  void SetUp() override {
    basisGate = std::get<0>(GetParam());
    eulerBases = std::get<1>(GetParam());
    target = std::get<2>(GetParam());
    targetDecomposition = TwoQubitWeylDecomposition::create(target, 1.0);
  }

  [[nodiscard]] static Eigen::Matrix4cd
  restore(const TwoQubitGateSequence& sequence) {
    Eigen::Matrix4cd matrix = Eigen::Matrix4cd::Identity();
    for (auto&& gate : sequence.gates) {
      matrix = getTwoQubitMatrix(gate) * matrix;
    }

    matrix *= helpers::globalPhaseFactor(sequence.globalPhase);
    return matrix;
  }

protected:
  Eigen::Matrix4cd target;
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
  auto stopTime = std::chrono::steady_clock::now() + std::chrono::seconds{3};
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
        testing::Values(Eigen::Matrix4cd::Identity(),
                        Eigen::kroneckerProduct(rzMatrix(1.0), ryMatrix(3.1)),
                        Eigen::kroneckerProduct(Eigen::Matrix2cd::Identity(),
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
                Eigen::kroneckerProduct(rxMatrix(1.0),
                                        Eigen::Matrix2cd::Identity()),
            Eigen::kroneckerProduct(rxMatrix(1.0), ryMatrix(1.0)) *
                canonicalGate(1.1, 0.2, 3.0) *
                Eigen::kroneckerProduct(rxMatrix(1.0),
                                        Eigen::Matrix2cd::Identity()),
            Eigen::kroneckerProduct(H_GATE, IPZ) *
                getTwoQubitMatrix(
                    {.type = qc::X, .parameter = {}, .qubitId = {0, 1}}) *
                Eigen::kroneckerProduct(IPX, IPY))));
