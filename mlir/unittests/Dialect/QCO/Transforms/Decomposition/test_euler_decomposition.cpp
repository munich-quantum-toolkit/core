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
#include "mlir/Dialect/QCO/Transforms/Decomposition/EulerBasis.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/EulerDecomposition.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateKind.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateSequence.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Helpers.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/UnitaryMatrices.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <optional>
#include <random>
#include <tuple>

using namespace mlir::qco;
using namespace mlir::qco::decomposition;
using namespace mlir::qco::decomposition_test;

static std::size_t countGatesOfType(const OneQubitGateSequence& seq,
                                    GateKind kind) {
  std::size_t count = 0;
  for (const auto& gate : seq.gates) {
    if (gate.type == kind) {
      ++count;
    }
  }
  return count;
}

/// Compare ``seq.getUnitaryMatrix()`` to ``u`` embedded on qubit 0 (4×4
/// layout).
static bool sequenceMatchesSingleQubitMatrix(const Eigen::Matrix2cd& u,
                                             const OneQubitGateSequence& seq,
                                             double tol = 1e-10) {
  const Eigen::Matrix4cd expanded = expandToTwoQubits(u, 0);
  return expanded.isApprox(seq.getUnitaryMatrix(), tol);
}

// NOLINTNEXTLINE(misc-use-internal-linkage) -- gtest `TEST_P` at global scope
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

protected:
  void SetUp() override {
    eulerBasis = std::get<0>(GetParam());
    originalMatrix = std::get<1>(GetParam())();
  }

  Eigen::Matrix2cd originalMatrix;
  EulerBasis eulerBasis{};
};

TEST_P(EulerDecompositionTest, TestExact) {
  auto decomposition = EulerDecomposition::generateCircuit(
      eulerBasis, originalMatrix, false, std::nullopt);
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
  std::size_t currentEulerBasis = 0;
  for (int i = 0; i < maxIterations; ++i) {
    auto originalMatrix = randomUnitaryMatrix<Eigen::Matrix2cd>(rng);
    auto eulerBasis = eulerBases[currentEulerBasis++ % eulerBases.size()];
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

TEST(EulerDecompositionTest, ZyzAnglesFromUnitaryReconstructHadamard) {
  Eigen::Matrix2cd hadamard;
  hadamard << 1.0 / std::numbers::sqrt2, 1.0 / std::numbers::sqrt2,
      1.0 / std::numbers::sqrt2, -1.0 / std::numbers::sqrt2;

  const auto angles =
      EulerDecomposition::anglesFromUnitary(hadamard, EulerBasis::ZYZ);
  const Eigen::Matrix2cd reconstructed =
      u3Matrix(angles[0], angles[1], angles[2]);

  EXPECT_TRUE(isEquivalentUpToGlobalPhase(hadamard, reconstructed));
}

TEST(EulerDecompositionTest, NativeEulerBasesRandomReconstruction) {
  std::mt19937 rng(424242);
  std::uniform_real_distribution<double> angleDist(-std::numbers::pi,
                                                   std::numbers::pi);
  for (int i = 0; i < 24; ++i) {
    const double theta = angleDist(rng);
    const double phi = angleDist(rng);
    const double lambda = angleDist(rng);
    const double phase = angleDist(rng);
    const Eigen::Matrix2cd unitary =
        std::exp(std::complex<double>(0.0, phase)) *
        u3Matrix(theta, phi, lambda);
    const Eigen::Matrix4cd expanded = expandToTwoQubits(unitary, 0);

    const auto u3Seq = EulerDecomposition::generateCircuit(
        EulerBasis::U3, unitary, true, std::nullopt);
    const auto zsxSeq = EulerDecomposition::generateCircuit(
        EulerBasis::ZSX, unitary, true, std::nullopt);
    const auto zsxxSeq = EulerDecomposition::generateCircuit(
        EulerBasis::ZSXX, unitary, true, std::nullopt);

    EXPECT_TRUE(
        isEquivalentUpToGlobalPhase(expanded, u3Seq.getUnitaryMatrix()));
    EXPECT_TRUE(
        isEquivalentUpToGlobalPhase(expanded, zsxSeq.getUnitaryMatrix()));
    EXPECT_TRUE(sequenceMatchesSingleQubitMatrix(unitary, zsxSeq));
    EXPECT_TRUE(sequenceMatchesSingleQubitMatrix(unitary, zsxxSeq));

    const std::size_t zsxSx = countGatesOfType(zsxSeq, GateKind::SX);
    const std::size_t zsxxSx = countGatesOfType(zsxxSeq, GateKind::SX);
    const std::size_t zsxxX = countGatesOfType(zsxxSeq, GateKind::X);
    EXPECT_EQ(countGatesOfType(zsxSeq, GateKind::X), 0U);
    EXPECT_LE(zsxxX, 1U);
    if (zsxxX == 0U) {
      EXPECT_EQ(zsxSx, zsxxSx);
    } else {
      EXPECT_EQ(zsxSx, zsxxSx + 2U);
    }
  }
}

TEST(EulerDecompositionTest, ZsxxPauliXUsesSingleXGate) {
  Eigen::Matrix2cd pauliX;
  pauliX << 0.0, 1.0, 1.0, 0.0;
  const auto seq = EulerDecomposition::generateCircuit(EulerBasis::ZSXX, pauliX,
                                                       true, std::nullopt);
  EXPECT_EQ(countGatesOfType(seq, GateKind::X), 1U);
  EXPECT_EQ(countGatesOfType(seq, GateKind::SX), 0U);
  EXPECT_TRUE(sequenceMatchesSingleQubitMatrix(pauliX, seq));
}

TEST(EulerDecompositionTest, GetGateTypesForEulerBasis) {
  const auto zyz = getGateTypesForEulerBasis(EulerBasis::ZYZ);
  ASSERT_EQ(zyz.size(), 2U);
  EXPECT_EQ(zyz[0], GateKind::RZ);
  EXPECT_EQ(zyz[1], GateKind::RY);

  const auto uFamily = getGateTypesForEulerBasis(EulerBasis::U321);
  ASSERT_EQ(uFamily.size(), 1U);
  EXPECT_EQ(uFamily[0], GateKind::U);

  const auto zsxx = getGateTypesForEulerBasis(EulerBasis::ZSXX);
  ASSERT_EQ(zsxx.size(), 3U);
  EXPECT_EQ(zsxx[0], GateKind::RZ);
  EXPECT_EQ(zsxx[1], GateKind::SX);
  EXPECT_EQ(zsxx[2], GateKind::X);
}

TEST(EulerDecompositionTest, UAndU321MatchU3Reconstruction) {
  std::mt19937 rng(99991);
  for (int i = 0; i < 32; ++i) {
    const auto u = randomUnitaryMatrix<Eigen::Matrix2cd>(rng);
    const auto seqU3 = EulerDecomposition::generateCircuit(EulerBasis::U3, u,
                                                           true, std::nullopt);
    const auto seqU = EulerDecomposition::generateCircuit(EulerBasis::U, u,
                                                          true, std::nullopt);
    const auto seqU321 = EulerDecomposition::generateCircuit(
        EulerBasis::U321, u, true, std::nullopt);
    EXPECT_TRUE(EulerDecompositionTest::restore(seqU3).isApprox(u));
    EXPECT_TRUE(EulerDecompositionTest::restore(seqU).isApprox(u));
    EXPECT_TRUE(EulerDecompositionTest::restore(seqU321).isApprox(u));
  }
}

TEST(EulerDecompositionTest, AnglesFromUnitaryXZXReconstructsRx) {
  const Eigen::Matrix2cd u = rxMatrix(0.7);
  (void)EulerDecomposition::anglesFromUnitary(u, EulerBasis::XZX);
  const auto seq = EulerDecomposition::generateCircuit(EulerBasis::XZX, u,
                                                       false, std::nullopt);
  EXPECT_TRUE(EulerDecompositionTest::restore(seq).isApprox(u));
}

TEST(EulerDecompositionTest, GateSequenceComplexityAndGlobalPhase) {
  OneQubitGateSequence seq;
  seq.gates.push_back(
      {.type = GateKind::RZ, .parameter = {0.2}, .qubitId = {0}});
  seq.globalPhase = 0.5;
  EXPECT_TRUE(seq.hasGlobalPhase());
  EXPECT_GE(seq.complexity(), 1U);
  seq.globalPhase = 0.0;
  EXPECT_FALSE(seq.hasGlobalPhase());
}

INSTANTIATE_TEST_SUITE_P(
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
