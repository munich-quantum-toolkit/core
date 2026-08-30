/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>

using namespace dd;

namespace {
/**
 * @brief Compare the elements of @p a and @p b with precision @p delta.
 */
void expectStateVectorNear(CVec a, CVec b, double delta = 1e-6) {
  for (std::size_t i = 0; i < b.size(); ++i) {
    EXPECT_NEAR(a[i].real(), b[i].real(), delta);
    EXPECT_NEAR(a[i].imag(), b[i].imag(), delta);
  }
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

  expectStateVectorNear(basis.getVector(), vec);

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

  expectStateVectorNear(ghz.getVector(), vec);

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

  const CVec vec{0,
                 std::numbers::inv_sqrt3,
                 std::numbers::inv_sqrt3,
                 0,
                 std::numbers::inv_sqrt3,
                 0,
                 0,
                 0};

  auto dd = std::make_unique<Package>(nq);
  auto w = makeWState(nq, *dd);

  expectStateVectorNear(w.getVector(), vec);

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
