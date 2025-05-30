/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>

using namespace dd;

namespace {
double norm(const std::vector<std::complex<double>>& v) {
  double sum{};
  for (const auto& entry : v) {
    sum += std::pow(std::abs(entry), 2);
  }
  return sum;
}
}; // namespace

///-----------------------------------------------------------------------------
///                      \n generate random VectorDDs \n
///-----------------------------------------------------------------------------

TEST(StateGenerationTest, OneQubit) {
  constexpr std::size_t nq = 1;

  const std::vector<std::size_t> nodesPerLevel{};

  auto dd = std::make_unique<Package>(nq);
  auto state = generateRandomState(nq, nodesPerLevel, ROUNDROBIN, *dd);

  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);
  EXPECT_EQ(state.size(), 2); // Node plus Terminal.

  dd->decRef(state);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, ExponentialState) {
  constexpr std::size_t nq = 3;

  auto dd = std::make_unique<Package>(nq);
  auto state = generateExponentialState(nq, *dd);

  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);
  EXPECT_EQ(state.size(), std::pow(2, nq));

  dd->decRef(state);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, ExponentialStateWithSeed) {
  constexpr std::size_t nq = 3;

  auto dd = std::make_unique<Package>(nq);
  auto state = generateExponentialState(nq, 42U, *dd);
  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);
  EXPECT_EQ(state.size(), std::pow(2, nq));

  dd->decRef(state);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, RandomStateStateRoundRobin) {
  constexpr std::size_t nq = 5;

  const std::vector<std::size_t> nodesPerLevel{2, 3, 4, 5};
  const std::size_t numNodes = (2 + 3 + 4 + 5) + 1; // plus root node.

  auto dd = std::make_unique<Package>(nq);
  auto state = generateRandomState(nq, nodesPerLevel, ROUNDROBIN, *dd);

  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);
  EXPECT_EQ(state.size(), numNodes + 1); // plus terminal.

  dd->decRef(state);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, RandomStateStateRandom) {
  constexpr std::size_t nq = 6;

  const std::vector<std::size_t> nodesPerLevel{2, 4, 8, 10, 12};

  auto dd = std::make_unique<Package>(nq);
  auto state = generateRandomState(nq, nodesPerLevel, RANDOM, *dd);

  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);

  dd->decRef(state);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, RandomStateStateRandomWithSeed) {
  constexpr std::size_t nq = 8;

  const std::vector<std::size_t> nodesPerLevel{2, 2, 2, 2, 2, 2, 2};

  auto dd = std::make_unique<Package>(nq);
  auto state = generateRandomState(nq, nodesPerLevel, RANDOM, 1337U, *dd);

  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);

  dd->decRef(state);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}
