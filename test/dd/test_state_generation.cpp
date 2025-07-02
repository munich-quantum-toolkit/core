/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "dd/StateGeneration.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace dd;

namespace {
/**
 * @brief Compare the elements of @p a and @p b with precision @p delta.
 */
void vecNear(CVec a, CVec b, double delta = 1e-6) {
  for (std::size_t i = 0; i < b.size(); ++i) {
    EXPECT_NEAR(a[i].real(), b[i].real(), delta);
    EXPECT_NEAR(a[i].imag(), b[i].imag(), delta);
  }
}

double norm(const std::vector<std::complex<double>>& v) {
  double sum{};
  for (const auto& entry : v) {
    sum += std::norm(entry);
  }
  return sum;
}
}; // namespace

///-----------------------------------------------------------------------------
///                      \n make VectorDDs \n
///-----------------------------------------------------------------------------

TEST(StateGenerationTest, MakeZero) {

  // Test: Produce valid zero state.
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 6;
  constexpr std::size_t len = 1ULL << nq;

  CVec vec(len);
  vec[0] = {1., 0};

  auto dd = std::make_unique<Package>(nq);
  auto zero = makeZeroState(nq, *dd);

  EXPECT_EQ(zero.getVector(), vec);

  dd->decRef(zero);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, MakeBasis) {

  // Test: Produce valid basis state.
  // Expect: |1011⟩ = [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]^T
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 4;
  constexpr std::size_t len = 1ULL << nq;

  const std::vector<bool> state{true, false, true, true};

  CVec vec(len);
  vec[13] = {1., 0};

  auto dd = std::make_unique<Package>(nq);
  auto basis = makeBasisState(nq, state, *dd);

  EXPECT_EQ(basis.getVector(), vec);

  dd->decRef(basis);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, MakeBasisDifficult) {

  // Test: Produce valid basis state.
  // Expect: |+⟩|-⟩|R⟩|L⟩ = (1/4)[1 1 -1 -1 i i -i -i -i -i i i 1 1 -1 -1]^T
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 4;

  const std::vector<BasisStates> state{BasisStates::plus, BasisStates::minus,
                                       BasisStates::right, BasisStates::left};

  const CVec vec{
      {.25, 0},  {.25, 0},  {-.25, 0}, {-.25, 0}, {0, .25}, {0, .25},
      {0, -.25}, {0, -.25}, {0, -.25}, {0, -.25}, {0, .25}, {0, .25},
      {.25, 0},  {.25, 0},  {-.25, 0}, {-.25, 0},
  };

  auto dd = std::make_unique<Package>(nq);
  auto basis = makeBasisState(nq, state, *dd);

  vecNear(basis.getVector(), vec);

  dd->decRef(basis);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, MakeGHZ) {

  // Test: Produce valid GHZ state.
  // Expect: 1/sqrt(2)(|0000⟩ + |1111⟩)
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 4;
  constexpr std::size_t len = 1ULL << nq;

  CVec vec(len);
  vec[0] = {SQRT2_2, 0};
  vec[len - 1] = {SQRT2_2, 0};

  auto dd = std::make_unique<Package>(nq);
  auto ghz = makeGHZState(nq, *dd);

  vecNear(ghz.getVector(), vec);

  dd->decRef(ghz);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, MakeGHZZeroQubits) {

  // Test: Produce valid GHZ state for zero qubits.
  // Expect: vEdge::one()

  constexpr std::size_t nq = 1;

  auto dd = std::make_unique<Package>(nq);
  auto ghz = makeGHZState(0, *dd);

  EXPECT_EQ(ghz, vEdge::one());
}

TEST(StateGenerationTest, MakeW) {

  // Test: Produce valid W state.
  // Expect: 1/sqrt(3)(|001⟩ + |010⟩ + |100⟩)
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 3;

  const CVec vec{
      0, 1 / std::sqrt(3), 1 / std::sqrt(3), 0, 1 / std::sqrt(3), 0, 0, 0};

  auto dd = std::make_unique<Package>(nq);
  auto w = makeWState(nq, *dd);

  vecNear(w.getVector(), vec);

  dd->decRef(w);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, MakeWZeroQubits) {

  // Test: Produce valid W state for zero qubits.
  // Expect: vEdge::one()

  constexpr std::size_t nq = 1;

  auto dd = std::make_unique<Package>(nq);
  auto w = makeWState(0, *dd);

  EXPECT_EQ(w, vEdge::one());
}

TEST(StateGenerationTest, FromVectorZero) {

  // Test: Return number zero on empty state vector.
  // Expect: Return vEdge::one()

  constexpr std::size_t nq = 1;

  const CVec vec{};

  auto dd = std::make_unique<Package>(nq);
  auto psi = makeStateFromVector(vec, *dd);

  EXPECT_EQ(psi, vEdge::one());
}

TEST(StateGenerationTest, FromVectorScalar) {

  // Test: Return scalar terminal for state vector of size 1.
  // Expect: vEdge::terminal(alpha)

  constexpr std::size_t nq = 1;
  constexpr std::complex<double> alpha{92., 2.};

  const CVec vec{alpha};

  auto dd = std::make_unique<Package>(nq);
  auto psi = makeStateFromVector(vec, *dd);

  EXPECT_TRUE(psi.isTerminal());
  EXPECT_TRUE(psi.w.approximatelyEquals(dd->cn.lookup(alpha)));
}

TEST(StateGenerationTest, FromVector) {

  // Test: Produce valid vector DD from state vector.
  // Expect: The Vector DD built from the state vector equals the directly
  //         constructed DD.
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 4;

  const CVec vec{
      {.25, 0},  {.25, 0},  {-.25, 0}, {-.25, 0}, {0, .25}, {0, .25},
      {0, -.25}, {0, -.25}, {0, -.25}, {0, -.25}, {0, .25}, {0, .25},
      {.25, 0},  {.25, 0},  {-.25, 0}, {-.25, 0},
  };

  const std::vector<BasisStates> state{BasisStates::plus, BasisStates::minus,
                                       BasisStates::right, BasisStates::left};

  auto dd = std::make_unique<Package>(nq);
  auto ref = makeBasisState(nq, state, *dd);
  auto psi = makeStateFromVector(vec, *dd);

  EXPECT_EQ(psi, ref);

  dd->decRef(ref);
  dd->decRef(psi);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, MakeZeroInvalidArguments) {

  // Test: Misconfigured package (# of qubits).

  constexpr std::size_t nq = 2;

  auto dd = std::make_unique<Package>(nq);
  EXPECT_THROW({ makeZeroState(nq + 1, *dd); }, std::invalid_argument);
}

TEST(StateGenerationTest, MakeBasisInvalidArguments) {

  // Test: Misconfigured package (# of qubits).
  // Test: Invalid size for `state` vector.

  constexpr std::size_t nq = 2;

  auto dd = std::make_unique<Package>(nq);
  const std::vector<BasisStates> state{BasisStates::one};

  EXPECT_THROW({ makeBasisState(nq + 1, state, *dd); }, std::invalid_argument);
  EXPECT_THROW({ makeBasisState(nq, state, *dd); }, std::invalid_argument);
}

TEST(StateGenerationTest, MakeGHZInvalidArguments) {

  // Test: Misconfigured package (# of qubits).

  constexpr std::size_t nq = 2;

  auto dd = std::make_unique<Package>(nq);
  EXPECT_THROW({ makeGHZState(nq + 1, *dd); }, std::invalid_argument);
}

TEST(StateGenerationTest, MakeWInvalidArguments) {

  // Test: Misconfigured package (# of qubits).

  constexpr std::size_t nq = 100;

  auto dd = std::make_unique<Package>(nq);
  EXPECT_THROW({ makeWState(nq + 1, *dd); }, std::invalid_argument);

  const auto tol = dd::RealNumber::eps;
  dd::ComplexNumbers::setTolerance(1);
  EXPECT_THROW({ makeWState(nq, *dd); }, std::invalid_argument);
  dd::ComplexNumbers::setTolerance(tol); // Reset tolerance.
}

TEST(StateGenerationTest, FromVectorInvalidArguments) {

  // Test: Misconfigured package (# of qubits).
  // Test: Invalid length of state vector.

  constexpr std::size_t nq = 2;

  auto dd = std::make_unique<Package>(nq);
  EXPECT_THROW({ makeStateFromVector(CVec(5), *dd); }, std::invalid_argument);
  EXPECT_THROW({ makeStateFromVector(CVec(3), *dd); }, std::invalid_argument);
}

///-----------------------------------------------------------------------------
///                      \n generate random VectorDDs \n
///-----------------------------------------------------------------------------

TEST(StateGenerationTest, GenerateExponential) {

  // Test: Generate a random exponentially large vector DD with a random seed.
  // Expect: The norm of the resulting vector DD must be 1.
  // Expect: The size of the resulting vector DD must be exponentially large.
  // Expect: If rebuild from a state vector the DDs are approximately the same.
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 3;

  const auto dd = std::make_unique<Package>(nq);
  const auto state = generateExponentialState(nq, *dd);
  const auto rebuild = makeStateFromVector(state.getVector(), *dd);
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

TEST(StateGenerationTest, GenerateExponentialWithSeed) {

  // Test: Generate a random exponentially large vector DD with a given seed.
  // Expect: The norm of the resulting vector DD must be 1.
  // Expect: The size of the resulting vector DD must be exponentially large.
  // Expect: If rebuild from a state vector the DDs are approximately the same.
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 3;

  const auto dd = std::make_unique<Package>(nq);
  const auto state = generateExponentialState(nq, *dd, 42U);
  const auto rebuild = makeStateFromVector(state.getVector(), *dd);
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

TEST(StateGenerationTest, GenerateRandomOneQubit) {

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

TEST(StateGenerationTest, GenerateRandomRoundRobin) {

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
  const auto rebuild = makeStateFromVector(state.getVector(), *dd);

  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);
  EXPECT_EQ(state.size(), size + 1); // plus terminal.
  EXPECT_EQ(state, rebuild);
  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), size);

  dd->decRef(state);
  dd->decRef(rebuild);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, GenerateRandomRoundRobinWithSeed) {

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
  const auto rebuild = makeStateFromVector(state.getVector(), *dd);

  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);
  EXPECT_EQ(state.size(), size + 1); // plus terminal.
  EXPECT_EQ(state, rebuild);
  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), size);

  dd->decRef(state);
  dd->decRef(rebuild);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, GenerateRandomRandom) {

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
  const auto rebuild = makeStateFromVector(state.getVector(), *dd);

  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);
  EXPECT_EQ(state.size(), size + 1); // plus terminal.
  EXPECT_EQ(state, rebuild);
  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), size);

  dd->decRef(state);
  dd->decRef(rebuild);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, GenerateRandomRandomWithSeed) {

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
  const auto rebuild = makeStateFromVector(state.getVector(), *dd);

  EXPECT_NEAR(norm(state.getVector()), 1., 1e-6);
  EXPECT_EQ(state.size(), size + 1); // plus terminal.
  EXPECT_EQ(state, rebuild);
  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), size);

  dd->decRef(state);
  dd->decRef(rebuild);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, GenerateRandomInvalidArguments) {

  // Test: Misconfigured package (# of qubits).
  // Test: Number of levels must be greater than zero.
  // Test: Invalid size of nodesPerLevel.
  // Test: Invalid nodesPerLevel.

  constexpr std::size_t nq = 3;

  auto dd = std::make_unique<Package>(nq);

  EXPECT_THROW(
      { generateRandomState(nq + 1, {}, RANDOM, *dd, 1337U); },
      std::invalid_argument);

  EXPECT_THROW(
      { generateRandomState(0, {0}, RANDOM, *dd, 1337U); },
      std::invalid_argument);

  EXPECT_THROW(
      { generateRandomState(nq, {0}, RANDOM, *dd, 1337U); },
      std::invalid_argument);

  EXPECT_THROW(
      { generateRandomState(nq, {1, 2, 5}, RANDOM, *dd, 1337U); },
      std::invalid_argument);
}
