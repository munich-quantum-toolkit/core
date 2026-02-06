/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Passes/Decomposition/EulerBasis.h"
#include "mlir/Passes/Decomposition/EulerDecomposition.h"
#include "mlir/Passes/Decomposition/GateSequence.h"
#include "mlir/Passes/Decomposition/Helpers.h"
#include "mlir/Passes/Decomposition/UnitaryMatrices.h"

#include <Eigen/QR>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <optional>
#include <tuple>

using namespace mlir::qco;
using namespace mlir::qco::decomposition;

namespace {
[[nodiscard]] Eigen::Matrix2cd randomUnitaryMatrix() {
  [[maybe_unused]] static auto initializeRandom = []() {
    // Eigen uses std::rand() internally, use fixed seed for deterministic
    // testing behavior
    std::srand(123456UL);
    return true;
  }();
  const Eigen::Matrix2cd randomMatrix = Eigen::Matrix2cd::Random();
  Eigen::HouseholderQR<Eigen::Matrix2cd> qr{}; // NOLINT(misc-include-cleaner)
  qr.compute(randomMatrix);
  const Eigen::Matrix2cd unitaryMatrix = qr.householderQ();
  assert(helpers::isUnitaryMatrix(unitaryMatrix));
  return unitaryMatrix;
}
} // namespace

class EulerDecompositionTest
    : public testing::TestWithParam<std::tuple<EulerBasis, Eigen::Matrix2cd>> {
public:
  [[nodiscard]] static Eigen::Matrix2cd
  restore(const TwoQubitGateSequence& sequence) {
    Eigen::Matrix2cd matrix = Eigen::Matrix2cd::Identity();
    for (auto&& gate : sequence.gates) {
      matrix = getSingleQubitMatrix(gate) * matrix;
    }

    matrix *= helpers::globalPhaseFactor(sequence.globalPhase);
    return matrix;
  }

  void SetUp() override {
    eulerBasis = std::get<0>(GetParam());
    originalMatrix = std::get<1>(GetParam());
  }

protected:
  Eigen::Matrix2cd originalMatrix;
  EulerBasis eulerBasis{};
};

TEST_P(EulerDecompositionTest, TestExact) {
  auto decomposition = EulerDecomposition::generateCircuit(
      eulerBasis, originalMatrix, false, 0.0);
  auto restoredMatrix = restore(decomposition);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix))
      << "RESULT:\n"
      << restoredMatrix << '\n';
}

TEST(EulerDecompositionTest, Random) {
  auto stopTime = std::chrono::steady_clock::now() + std::chrono::seconds{3};
  auto iterations = 0;
  auto eulerBases = std::array{EulerBasis::XYX, EulerBasis::XZX,
                               EulerBasis::ZYZ, EulerBasis::ZXZ};
  std::size_t currentEulerBase = 0;
  while (std::chrono::steady_clock::now() < stopTime) {
    auto originalMatrix = randomUnitaryMatrix();
    auto eulerBasis = eulerBases[currentEulerBase++ % eulerBases.size()];
    auto decomposition = EulerDecomposition::generateCircuit(
        eulerBasis, originalMatrix, true, std::nullopt);
    auto restoredMatrix = EulerDecompositionTest::restore(decomposition);

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
    SingleQubitMatrices, EulerDecompositionTest,
    testing::Combine(testing::Values(EulerBasis::XYX, EulerBasis::XZX,
                                     EulerBasis::ZYZ, EulerBasis::ZXZ),
                     testing::Values(Eigen::Matrix2cd::Identity(),
                                     ryMatrix(2.0), rxMatrix(0.5),
                                     rzMatrix(3.14), H_GATE)));
