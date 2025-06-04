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

#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace dd;

namespace {
double norm(const std::vector<std::complex<double>>& v) {
  double sum{};
  for (const auto& entry : v) {
    sum += std::norm(entry);
  }
  return sum;
}
}; // namespace

///-----------------------------------------------------------------------------
///                      \n generate random VectorDDs \n
///-----------------------------------------------------------------------------

TEST(StateGenerationTest, OneQubit) {

  // Test: Generate a random single qubit vector DD.
  // Expect: The norm of the resulting vector DD must be 1.
  // Expect: The size of the resulting vector DD must be 2 (node + terminal).
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 1;

  const std::vector<std::size_t> nodesPerLevel{1};

  const auto dd = std::make_unique<Package>(nq);
  const auto state = generateRandomState(nq, nodesPerLevel, ROUNDROBIN, *dd);

  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);
  EXPECT_EQ(state.size(), 2); // Node plus Terminal.
  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 1);

  dd->decRef(state);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, ExponentialState) {

  // Test: Generate a random exponentially large vector DD with a random seed.
  // Expect: The norm of the resulting vector DD must be 1.
  // Expect: The size of the resulting vector DD must be exponentially large.
  // Expect: If rebuild from a state vector the DDs are approximately the same.
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 3;

  const auto dd = std::make_unique<Package>(nq);
  const auto state = generateExponentialState(nq, *dd);
  const auto rebuild = dd->makeStateFromVector(state.getVector());
  const std::size_t size = 1ULL << nq;

  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);
  EXPECT_EQ(state.size(), size);
  EXPECT_EQ(state, rebuild);
  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), size - 1);

  dd->decRef(state);
  dd->decRef(rebuild);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, ExponentialStateWithSeed) {

  // Test: Generate a random exponentially large vector DD with a given seed.
  // Expect: The norm of the resulting vector DD must be 1.
  // Expect: The size of the resulting vector DD must be exponentially large.
  // Expect: If rebuild from a state vector the DDs are approximately the same.
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 3;

  const auto dd = std::make_unique<Package>(nq);
  const auto state = generateExponentialState(nq, *dd, 42U);
  const auto rebuild = dd->makeStateFromVector(state.getVector());
  const std::size_t size = 1ULL << nq;

  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);
  EXPECT_EQ(state.size(), size);
  EXPECT_EQ(state, rebuild);
  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), size - 1);

  dd->decRef(state);
  dd->decRef(rebuild);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, RandomStateRoundRobin) {

  // Test: Generate a random vector DD using the round-robin strategy.
  // Expect: The norm of the resulting vector DD must be 1.
  // Expect: The size of the resulting vector DD must be the sum of the
  //         specified nodes.
  // Expect: If rebuild from a state vector the DDs are approximately the same.
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 5;

  const std::vector<std::size_t> nodesPerLevel{1, 2, 3, 4, 5};
  const std::size_t size =
      std::accumulate(nodesPerLevel.begin(), nodesPerLevel.end(), 0UL);

  const auto dd = std::make_unique<Package>(nq);
  const auto state = generateRandomState(nq, nodesPerLevel, ROUNDROBIN, *dd);
  const auto rebuild = dd->makeStateFromVector(state.getVector());

  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);
  EXPECT_EQ(state.size(), size + 1); // plus terminal.
  EXPECT_EQ(state, rebuild);
  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), size);

  dd->decRef(state);
  dd->decRef(rebuild);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, RandomStateRoundRobinWithSeed) {

  // Test: Generate a random vector DD using the round-robin strategy with a
  //       given seed.
  // Expect: The norm of the resulting vector DD must be 1.
  // Expect: The size of the resulting vector DD must be the sum of the
  //         specified nodes.
  // Expect: If rebuild from a state vector the DDs are approximately the same.
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 5;

  const std::vector<std::size_t> nodesPerLevel{1, 2, 3, 4, 5};
  const std::size_t size =
      std::accumulate(nodesPerLevel.begin(), nodesPerLevel.end(), 0UL);

  const auto dd = std::make_unique<Package>(nq);
  const auto state =
      generateRandomState(nq, nodesPerLevel, ROUNDROBIN, *dd, 72U);
  const auto rebuild = dd->makeStateFromVector(state.getVector());

  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);
  EXPECT_EQ(state.size(), size + 1); // plus terminal.
  EXPECT_EQ(state, rebuild);
  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), size);

  dd->decRef(state);
  dd->decRef(rebuild);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, RandomStateRandom) {

  // Test: Generate a random vector DD using the random strategy.
  // Expect: The norm of the resulting vector DD must be 1.
  // Expect: The size of the resulting vector DD must be the sum of the
  //         specified nodes.
  // Expect: If rebuild from a state vector the DDs are approximately the same.
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 6;

  const std::vector<std::size_t> nodesPerLevel{1, 2, 4, 8, 10, 12};
  const std::size_t size =
      std::accumulate(nodesPerLevel.begin(), nodesPerLevel.end(), 0UL);

  const auto dd = std::make_unique<Package>(nq);
  const auto state = generateRandomState(nq, nodesPerLevel, RANDOM, *dd);
  const auto rebuild = dd->makeStateFromVector(state.getVector());

  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);
  EXPECT_EQ(state.size(), size + 1); // plus terminal.
  EXPECT_EQ(state, rebuild);
  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), size);

  dd->decRef(state);
  dd->decRef(rebuild);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, RandomStateRandomWithSeed) {

  // Test: Generate a random vector DD using the random strategy with a given
  //       seed.
  // Expect: The norm of the resulting vector DD must be 1.
  // Expect: The size of the resulting vector DD must be the sum of the
  //         specified nodes.
  // Expect: If rebuild from a state vector the DDs are approximately the same.
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 8;

  const std::vector<std::size_t> nodesPerLevel{1, 2, 2, 2, 2, 2, 2, 2};
  const std::size_t size =
      std::accumulate(nodesPerLevel.begin(), nodesPerLevel.end(), 0UL);

  const auto dd = std::make_unique<Package>(nq);
  const auto state = generateRandomState(nq, nodesPerLevel, RANDOM, *dd, 1337U);
  const auto rebuild = dd->makeStateFromVector(state.getVector());

  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);
  EXPECT_EQ(state.size(), size + 1); // plus terminal.
  EXPECT_EQ(state, rebuild);
  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), size);

  dd->decRef(state);
  dd->decRef(rebuild);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, InvalidLevels) {

  // Test: Number of levels must be greater than zero.

  constexpr std::size_t nq = 0;

  const std::vector<std::size_t> nodesPerLevel{};
  auto dd = std::make_unique<Package>(nq);
  EXPECT_THROW(
      { generateRandomState(nq, nodesPerLevel, RANDOM, *dd, 1337U); },
      std::invalid_argument);
}

TEST(StateGenerationTest, InvalidNodesPerLevelSize) {

  // Test: Invalid size of nodesPerLevel.

  constexpr std::size_t nq = 3;

  const std::vector<std::size_t> nodesPerLevel{1, 2, 3, 4};
  auto dd = std::make_unique<Package>(nq);
  EXPECT_THROW(
      { generateRandomState(nq, nodesPerLevel, RANDOM, *dd, 1337U); },
      std::invalid_argument);
}

TEST(StateGenerationTest, InvalidNodesPerLevel) {

  // Test: Invalid nodesPerLevel.

  constexpr std::size_t nq = 4;

  const std::vector<std::size_t> nodesPerLevel{1, 2, 4, 9};
  auto dd = std::make_unique<Package>(nq);
  EXPECT_THROW(
      { generateRandomState(nq, nodesPerLevel, RANDOM, *dd, 1337U); },
      std::invalid_argument);
}
