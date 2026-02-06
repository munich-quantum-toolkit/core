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

    matrix *= std::exp(C_IM * sequence.globalPhase);
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

TEST(BasisDecomposerTest, Crash) {
  using namespace std::complex_literals;
  matrix4x4 originalMatrix{{-0.23104450537689214 + -0.44268488901708902i,
                            -0.60656504798621003 + -0.27756198294119977i,
                            -0.2168858251642842 + -0.27845819247692827i,
                            -0.42430958159720128 + 0.032705758031399738i},
                           {0.12891961731437976 + -0.2577139933400836i,
                            0.059033561840284507 + 0.051774294297249751i,
                            0.58205201943239671 + -0.20399736896613216i,
                            -0.027126130902642431 + 0.72777907642808048i},
                           {0.60297884333102469 + 0.35765188950245741i,
                            -0.59087990913607613 + 0.22062558535485413i,
                            0.077633311362340196 + -0.28085102069787549i,
                            0.13707024540895657 + -0.083632801340620747i},
                           {-0.22282692851707037 + 0.3556143358154254i,
                            0.3014101239694616 + 0.24537699657586307i,
                            0.10077337125619777 + -0.63242595993554396i,
                            -0.48930698747658014 + -0.1526088977163027i}};

  const Gate basisGate{.type = qc::X, .parameter = {}, .qubitId = {0, 1}};
  const llvm::SmallVector<EulerBasis> eulerBases = {EulerBasis::XYX,
                                                    EulerBasis::ZXZ};

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
        testing::Values(matrix4x4::Identity(),
                        Eigen::kroneckerProduct(rzMatrix(1.0), ryMatrix(3.1)),
                        Eigen::kroneckerProduct(matrix2x2::Identity(),
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
                Eigen::kroneckerProduct(rxMatrix(1.0), matrix2x2::Identity()),
            Eigen::kroneckerProduct(rxMatrix(1.0), ryMatrix(1.0)) *
                canonicalGate(1.1, 0.2, 3.0) *
                Eigen::kroneckerProduct(rxMatrix(1.0), matrix2x2::Identity()),
            Eigen::kroneckerProduct(H_GATE, IPZ) *
                getTwoQubitMatrix(
                    {.type = qc::X, .parameter = {}, .qubitId = {0, 1}}) *
                Eigen::kroneckerProduct(IPX, IPY))));
