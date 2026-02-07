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
#include "utils.h"

#include <Eigen/QR>
#include <cassert>
#include <cstdlib>
#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <optional>
#include <random>
#include <tuple>
#include <unsupported/Eigen/KroneckerProduct>

using namespace mlir::qco;
using namespace mlir::qco::decomposition;

class BasisDecomposerTest
    : public testing::TestWithParam<std::tuple<
          Gate, llvm::SmallVector<EulerBasis>, Eigen::Matrix4cd (*)()>> {
public:
  void SetUp() override {
    basisGate = std::get<0>(GetParam());
    eulerBases = std::get<1>(GetParam());
    target = std::get<2>(GetParam())();
    targetDecomposition = std::make_unique<TwoQubitWeylDecomposition>(
        TwoQubitWeylDecomposition::create(target, 1.0));
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
  std::unique_ptr<TwoQubitWeylDecomposition> targetDecomposition;
};

TEST_P(BasisDecomposerTest, TestExact) {
  const auto& originalMatrix = target;
  auto decomposer = TwoQubitBasisDecomposer::create(basisGate, 1.0);
  auto decomposedSequence = decomposer.twoQubitDecompose(
      *targetDecomposition, eulerBases, 1.0, false, std::nullopt);

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
      *targetDecomposition, eulerBases, 1.0 - 1e-12, true, std::nullopt);

  ASSERT_TRUE(decomposedSequence.has_value());

  auto restoredMatrix = restore(*decomposedSequence);

  EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix))
      << "RESULT:\n"
      << restoredMatrix << '\n';
}

TEST(BasisDecomposerTest, Random) {
  constexpr auto maxIterations = 2000;

  const llvm::SmallVector<Gate, 2> basisGates{
      {.type = qc::X, .parameter = {}, .qubitId = {0, 1}},
      {.type = qc::X, .parameter = {}, .qubitId = {1, 0}}};
  const llvm::SmallVector<EulerBasis, 4> eulerBases = {
      EulerBasis::XYX, EulerBasis::ZXZ, EulerBasis::ZYZ, EulerBasis::XZX};
  std::mt19937 rng{123456UL};
  std::uniform_int_distribution<std::size_t> distBasisGate{
      0, basisGates.size() - 1};
  std::uniform_int_distribution<std::size_t> distEulerBases{
      1, eulerBases.size() - 1};

  auto selectRandomEulerBases = [&]() {
    auto tmp = eulerBases;
    llvm::shuffle(tmp.begin(), tmp.end(), rng);
    tmp.resize(distEulerBases(rng));
    return tmp;
  };
  auto selectRandomBasisGate = [&]() { return basisGates[distBasisGate(rng)]; };

  for (int i = 0; i < maxIterations; ++i) {
    auto originalMatrix = randomUnitaryMatrix<Eigen::Matrix4cd>();

    auto targetDecomposition =
        TwoQubitWeylDecomposition::create(originalMatrix, 1.0);
    auto decomposer =
        TwoQubitBasisDecomposer::create(selectRandomBasisGate(), 1.0);
    auto decomposedSequence = decomposer.twoQubitDecompose(
        targetDecomposition, selectRandomEulerBases(), 1.0, true, std::nullopt);

    ASSERT_TRUE(decomposedSequence.has_value());

    auto restoredMatrix = BasisDecomposerTest::restore(*decomposedSequence);

    EXPECT_TRUE(restoredMatrix.isApprox(originalMatrix))
        << "ORIGINAL:\n"
        << originalMatrix << '\n'
        << "RESULT:\n"
        << restoredMatrix << '\n';
  }
}

INSTANTIATE_TEST_SUITE_P(
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
        testing::Values(
            []() -> Eigen::Matrix4cd { return Eigen::Matrix4cd::Identity(); },
            []() -> Eigen::Matrix4cd {
              return Eigen::kroneckerProduct(rzMatrix(1.0), ryMatrix(3.1));
            },
            []() -> Eigen::Matrix4cd {
              return Eigen::kroneckerProduct(Eigen::Matrix2cd::Identity(),
                                             rxMatrix(0.1));
            })));

INSTANTIATE_TEST_SUITE_P(
    TwoQubitMatrices, BasisDecomposerTest,
    testing::Combine(
        // basis gates
        testing::Values(Gate{.type = qc::X, .parameter = {}, .qubitId = {0, 1}},
                        Gate{
                            .type = qc::X, .parameter = {}, .qubitId = {1, 0}}),
        // sets of euler bases
        testing::Values(
            llvm::SmallVector<EulerBasis>{EulerBasis::ZYZ},
            llvm::SmallVector<EulerBasis>{EulerBasis::ZYZ, EulerBasis::ZXZ,
                                          EulerBasis::XYX, EulerBasis::XZX},
            llvm::SmallVector<EulerBasis>{EulerBasis::XZX, EulerBasis::XYX}),
        // targets to be decomposed
        ::testing::Values(
            []() -> Eigen::Matrix4cd { return rzzMatrix(2.0); },
            []() -> Eigen::Matrix4cd {
              return ryyMatrix(1.0) * rzzMatrix(3.0) * rxxMatrix(2.0);
            },
            []() -> Eigen::Matrix4cd {
              return TwoQubitWeylDecomposition::getCanonicalMatrix(1.5, -0.2,
                                                                   0.0) *
                     Eigen::kroneckerProduct(rxMatrix(1.0),
                                             Eigen::Matrix2cd::Identity());
            },
            []() -> Eigen::Matrix4cd {
              return Eigen::kroneckerProduct(rxMatrix(1.0), ryMatrix(1.0)) *
                     TwoQubitWeylDecomposition::getCanonicalMatrix(1.1, 0.2,
                                                                   3.0) *
                     Eigen::kroneckerProduct(rxMatrix(1.0),
                                             Eigen::Matrix2cd::Identity());
            },
            []() -> Eigen::Matrix4cd {
              return Eigen::kroneckerProduct(H_GATE, IPZ) *
                     getTwoQubitMatrix(
                         {.type = qc::X, .parameter = {}, .qubitId = {0, 1}}) *
                     Eigen::kroneckerProduct(IPX, IPY);
            })));
