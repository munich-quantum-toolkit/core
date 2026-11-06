/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/ComplexNumbers.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/MatrixComparison.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"

#include "support/TestSupport.hpp"

#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <numbers>
#include <random>

using mqt::test::value;

namespace dd {
namespace {
TEST(MatrixComparison, IndependentDenseMatrices) {
  std::mt19937_64 rng(1134);
  std::uniform_real_distribution<double> random(-1., 1.);
  for (const auto qubits : {1U, 2U, 3U, 4U}) {
    const auto dimension = size_t{1} << qubits;
    for (size_t sample = 0; sample < 40; ++sample) {
      SCOPED_TRACE(qubits);
      SCOPED_TRACE(sample);
      auto package = value(Package::create(qubits));
      CMat matrix(dimension, CVec(dimension));
      std::complex<double> trace{};
      for (size_t i = 0; i < dimension; ++i) {
        for (size_t j = 0; j < dimension; ++j) {
          matrix[i][j] = std::complex<double>{random(rng), random(rng)} *
                         (sample < 20 ? 1. : 1e-8);
          if (i == j) {
            matrix[i][j] += sample < 20 ? 0. : 1.;
            trace += matrix[i][j];
          }
        }
      }
      const auto phase = std::abs(trace) == 0. ? std::complex<double>{1., 0.}
                                               : trace / std::abs(trace);
      const auto measured = size_t{1} << (sample % qubits);
      llvm::SmallVector<bool> garbage(qubits, true);
      garbage[sample % qubits] = false;
      double fixed = 0.;
      double projective = 0.;
      double leakage = 0.;
      for (size_t i = 0; i < dimension; ++i) {
        for (size_t j = 0; j < dimension; ++j) {
          fixed =
              std::hypot(fixed, std::abs(matrix[i][j] - (i == j ? 1. : 0.)));
          projective = std::hypot(
              projective, std::abs(matrix[i][j] - (i == j ? phase : 0.)));
          if (((i ^ j) & measured) != 0) {
            leakage = std::hypot(leakage, std::abs(matrix[i][j]));
          }
        }
      }
      const auto scale = std::sqrt(static_cast<double>(dimension));
      const auto edge = value(package->makeDDFromMatrix(matrix));
      for (const auto factor : {0.5, 2.}) {
        const auto fixedResult =
            compareToIdentity(edge, factor * fixed / scale);
        EXPECT_EQ(fixedResult.closeToIdentity, factor > 1.);
        const auto phaseResult =
            compareToIdentity(edge, factor * projective / scale);
        EXPECT_EQ(phaseResult.closeUpToGlobalPhase, factor > 1.);
        EXPECT_EQ(isCloseToDiagonal(edge, factor * leakage / scale, garbage),
                  factor > 1.);
      }
    }
  }
}

TEST(MatrixComparison, PhaseRootWeightAndProjector) {
  auto package = value(Package::create(1));
  auto scalar = mEdge::one();
  scalar.w = package->cn.lookup(0., 1.);
  EXPECT_FALSE(compareToIdentity(scalar).closeToIdentity);
  EXPECT_TRUE(compareToIdentity(scalar).closeUpToGlobalPhase);
  scalar.w = package->cn.lookup(0., 2.);
  EXPECT_FALSE(compareToIdentity(scalar).closeUpToGlobalPhase);
  EXPECT_FALSE(compareToIdentity(mEdge::zero()).closeUpToGlobalPhase);
  const auto projector =
      value(package->makeDDFromMatrix(CMat{{1., 0.}, {0., 0.}}));
  EXPECT_FALSE(compareToIdentity(projector, 0.7).closeUpToGlobalPhase);
  EXPECT_TRUE(compareToIdentity(projector, 0.71).closeUpToGlobalPhase);
}

TEST(MatrixComparison, WeightedSharedBlocksAndThresholdBoundary) {
  auto package = value(Package::create(60));
  auto block = value(package->makeDDFromMatrix(CMat{{0., 1.}, {1., 0.}}));
  block.w = package->cn.lookup(1e-4);
  auto other = block;
  other.w = package->cn.lookup(2e-4);
  const auto edge = package->makeDDNode(59, std::array{
                                                mEdge::one(),
                                                block,
                                                other,
                                                mEdge::one(),
                                            });
  const auto distance = std::sqrt(2.5) * 1e-4;
  EXPECT_TRUE(compareToIdentity(edge, distance * 1.01).closeToIdentity);
  EXPECT_FALSE(compareToIdentity(edge, distance * 0.99).closeUpToGlobalPhase);
  EXPECT_TRUE(isCloseToDiagonal(edge, distance * 1.01));
  EXPECT_FALSE(isCloseToDiagonal(edge, distance * 0.99));
  const auto triangular =
      value(package->makeDDFromMatrix(CMat{{1., 1e-4}, {0., 1.}}));
  const auto boundary = 1e-4 / std::numbers::sqrt2;
  EXPECT_TRUE(compareToIdentity(triangular, boundary).closeToIdentity);
  EXPECT_TRUE(isCloseToDiagonal(triangular, boundary));
  EXPECT_FALSE(
      compareToIdentity(triangular, boundary * 0.99).closeUpToGlobalPhase);
}

TEST(MatrixComparison, MeasuredAndGarbageBlocks) {
  auto package = value(Package::create(60));
  const auto flip = package->makeDDNode(59, std::array{
                                                mEdge::zero(),
                                                mEdge::one(),
                                                mEdge::one(),
                                                mEdge::zero(),
                                            });
  EXPECT_FALSE(isCloseToDiagonal(flip));
  llvm::SmallVector<bool> garbage(60);
  garbage[59] = true;
  EXPECT_TRUE(isCloseToDiagonal(flip, 1e-10, garbage));
  const auto phase =
      value(package->makeDDFromMatrix(CMat{{1., 0.}, {0., -1.}}));
  EXPECT_TRUE(isCloseToDiagonal(phase));
  EXPECT_FALSE(compareToIdentity(phase).closeUpToGlobalPhase);
}

TEST(MatrixComparison, TinyResiduals) {
  auto package = value(Package::create(1));
  for (const auto angle : {1e-4, 1e-8, 1e-10, 1e-12}) {
    const auto cosine = std::cos(angle / 2.);
    const std::complex<double> sine{0., -std::sin(angle / 2.)};
    const auto rotation =
        value(package->makeDDFromMatrix(CMat{{cosine, sine}, {sine, cosine}}));
    const auto distance = 2. * std::sin(angle / 4.);
    EXPECT_TRUE(compareToIdentity(rotation, distance * 1.01).closeToIdentity);
    EXPECT_FALSE(
        compareToIdentity(rotation, distance * 0.99).closeUpToGlobalPhase);
  }
  const auto oldTolerance = RealNumber::eps;
  value(ComplexNumbers::setTolerance(1e-250));
  const auto tiny = value(
      package->makeDDFromMatrix(CMat{{{1., 1e-200}, 0.}, {0., {1., -1e-200}}}));
  EXPECT_TRUE(compareToIdentity(tiny, 1.1e-200).closeToIdentity);
  EXPECT_FALSE(compareToIdentity(tiny, 0.9e-200).closeUpToGlobalPhase);
  value(ComplexNumbers::setTolerance(oldTolerance));
}
} // namespace
} // namespace dd
