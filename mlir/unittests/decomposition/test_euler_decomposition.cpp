/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/Decomposition/EulerBasis.h"
#include "mlir/Dialect/MQTOpt/Transforms/Decomposition/EulerDecomposition.h"
#include "mlir/Dialect/MQTOpt/Transforms/Decomposition/GateSequence.h"
#include "mlir/Dialect/MQTOpt/Transforms/Decomposition/Helpers.h"
#include "mlir/Dialect/MQTOpt/Transforms/Decomposition/UnitaryMatrices.h"

#include <Eigen/QR>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <optional>
#include <tuple>

using namespace mqt::ir::opt;
using namespace mqt::ir::opt::decomposition;

namespace {
[[nodiscard]] matrix2x2 randomUnitaryMatrix() {
  [[maybe_unused]] static auto initializeRandom = []() {
    // Eigen uses std::rand() internally, use fixed seed for deterministic
    // testing behavior
    std::srand(123456UL);
    return true;
  }();
  const matrix2x2 randomMatrix = matrix2x2::Random();
  Eigen::HouseholderQR<matrix2x2> qr{}; // NOLINT(misc-include-cleaner)
  qr.compute(randomMatrix);
  const matrix2x2 unitaryMatrix = qr.householderQ();
  assert(helpers::isUnitaryMatrix(unitaryMatrix));
  return unitaryMatrix;
}
} // namespace

class EulerDecompositionTest
    : public testing::TestWithParam<std::tuple<EulerBasis, matrix2x2>> {
public:
  [[nodiscard]] static matrix2x2 restore(const TwoQubitGateSequence& sequence) {
    matrix2x2 matrix = matrix2x2::Identity();
    for (auto&& gate : sequence.gates) {
      matrix = getSingleQubitMatrix(gate) * matrix;
    }

    matrix *= std::exp(IM * sequence.globalPhase);
    return matrix;
  }

  void SetUp() override {
    eulerBasis = std::get<0>(GetParam());
    originalMatrix = std::get<1>(GetParam());
  }

protected:
  matrix2x2 originalMatrix;
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
  auto stopTime = std::chrono::steady_clock::now() + std::chrono::seconds{10};
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
                     testing::Values(IDENTITY_GATE, ryMatrix(2.0),
                                     rxMatrix(0.5), rzMatrix(3.14), H_GATE)));
