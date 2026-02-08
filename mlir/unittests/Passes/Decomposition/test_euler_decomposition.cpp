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
#include "utils.h"

#include <Eigen/Core>
#include <Eigen/QR>
#include <array>
#include <cassert>
#include <cstdlib>
#include <gtest/gtest.h>
#include <optional>
#include <tuple>

using namespace mlir::qco;
using namespace mlir::qco::decomposition;

class EulerDecompositionTest
    : public testing::TestWithParam<
          std::tuple<EulerBasis, Eigen::Matrix2cd (*)()>> {
public:
  [[nodiscard]] static Eigen::Matrix2cd
  restore(const OneQubitGateSequence& sequence) {
    Eigen::Matrix2cd matrix = Eigen::Matrix2cd::Identity();
    for (auto&& gate : sequence.gates) {
      matrix = getSingleQubitMatrix(gate) * matrix;
    }

    matrix *= helpers::globalPhaseFactor(sequence.globalPhase);
    return matrix;
  }

  void SetUp() override {
    eulerBasis = std::get<0>(GetParam());
    originalMatrix = std::get<1>(GetParam())();
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
  constexpr auto maxIterations = 10000;
  std::mt19937 rng{12345678UL};

  auto eulerBases = std::array{EulerBasis::XYX, EulerBasis::XZX,
                               EulerBasis::ZYZ, EulerBasis::ZXZ};
  std::size_t currentEulerBase = 0;
  for (int i = 0; i < maxIterations; ++i) {
    auto originalMatrix = randomUnitaryMatrix<Eigen::Matrix2cd>(rng);
    auto eulerBasis = eulerBases[currentEulerBase++ % eulerBases.size()];
    auto decomposition = EulerDecomposition::generateCircuit(
        eulerBasis, originalMatrix, true, std::nullopt);
    auto restoredMatrix = EulerDecompositionTest::restore(decomposition);

    EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix))
        << "ORIGINAL:\n"
        << originalMatrix << '\n'
        << "RESULT:\n"
        << restoredMatrix << '\n';
  }
}

INSTANTIATE_TEST_CASE_P(
    SingleQubitMatrices, EulerDecompositionTest,
    testing::Combine(testing::Values(EulerBasis::XYX, EulerBasis::XZX,
                                     EulerBasis::ZYZ, EulerBasis::ZXZ),
                     testing::Values(
                         []() -> Eigen::Matrix2cd {
                           return Eigen::Matrix2cd::Identity();
                         },
                         []() -> Eigen::Matrix2cd { return ryMatrix(2.0); },
                         []() -> Eigen::Matrix2cd { return rxMatrix(0.5); },
                         []() -> Eigen::Matrix2cd { return rzMatrix(3.14); },
                         []() -> Eigen::Matrix2cd { return H_GATE; })));
