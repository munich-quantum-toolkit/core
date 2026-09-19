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
#include "dd/Error.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "dd/StateGeneration.hpp"

#include "TestUtils.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <numbers>
#include <vector>

using namespace dd;

namespace {
/// Compare the elements of @p a and @p b with precision @p delta.
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

  auto dd = test::value(Package::create(nq));
  auto const zero = test::value(makeZeroState(nq, *dd));

  EXPECT_EQ(zero.getVector(), vec);

  test::value(dd->decRef(zero));
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

  auto dd = test::value(Package::create(nq));
  auto const basis = test::value(makeBasisState(nq, state, *dd));

  EXPECT_EQ(basis.getVector(), vec);

  test::value(dd->decRef(basis));
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, MakeBasisDifficult) {

  // Test: Produce valid basis state.
  // Expect: |+⟩|-⟩|R⟩|L⟩ = (1/4)[1 1 -1 -1 i i -i -i -i -i i i 1 1 -1 -1]^T
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 4;

  const std::vector<BasisStates> state{
      BasisStates::plus,
      BasisStates::minus,
      BasisStates::right,
      BasisStates::left,
  };

  const CVec vec{
      {.25, 0},  {.25, 0},  {-.25, 0}, {-.25, 0}, {0, .25}, {0, .25},
      {0, -.25}, {0, -.25}, {0, -.25}, {0, -.25}, {0, .25}, {0, .25},
      {.25, 0},  {.25, 0},  {-.25, 0}, {-.25, 0},
  };

  auto dd = test::value(Package::create(nq));
  auto const basis = test::value(makeBasisState(nq, state, *dd));

  expectStateVectorNear(basis.getVector(), vec);

  test::value(dd->decRef(basis));
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

  auto dd = test::value(Package::create(nq));
  auto const ghz = test::value(makeGHZState(nq, *dd));

  expectStateVectorNear(ghz.getVector(), vec);

  test::value(dd->decRef(ghz));
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, MakeGHZZeroQubits) {

  // Test: Produce valid GHZ state for zero qubits.
  // Expect: vEdge::one()

  constexpr std::size_t nq = 1;

  auto dd = test::value(Package::create(nq));
  auto const ghz = test::value(makeGHZState(0, *dd));

  EXPECT_EQ(ghz, vEdge::one());
}

TEST(StateGenerationTest, MakeW) {

  // Test: Produce valid W state.
  // Expect: 1/sqrt(3)(|001⟩ + |010⟩ + |100⟩)
  // Expect: Properly increase and decrease the ref counts.

  constexpr std::size_t nq = 3;

  const CVec vec{
      0,
      std::numbers::inv_sqrt3,
      std::numbers::inv_sqrt3,
      0,
      std::numbers::inv_sqrt3,
      0,
      0,
      0,
  };

  auto dd = test::value(Package::create(nq));
  auto const w = test::value(makeWState(nq, *dd));

  expectStateVectorNear(w.getVector(), vec);

  test::value(dd->decRef(w));
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, MakeWZeroQubits) {

  // Test: Produce valid W state for zero qubits.
  // Expect: vEdge::one()

  constexpr std::size_t nq = 1;

  auto dd = test::value(Package::create(nq));
  auto const w = test::value(makeWState(0, *dd));

  EXPECT_EQ(w, vEdge::one());
}

TEST(StateGenerationTest, FromVectorZero) {

  // Test: Return number zero on empty state vector.
  // Expect: Return vEdge::one()

  constexpr std::size_t nq = 1;

  const CVec vec{};

  auto dd = test::value(Package::create(nq));
  auto const psi = test::value(makeStateFromVector(vec, *dd));

  EXPECT_EQ(psi, vEdge::one());
}

TEST(StateGenerationTest, FromVectorScalar) {

  // Test: Return scalar terminal for state vector of size 1.
  // Expect: vEdge::terminal(alpha)

  constexpr std::size_t nq = 1;
  constexpr std::complex<double> alpha{92., 2.};

  const CVec vec{alpha};

  auto dd = test::value(Package::create(nq));
  auto const psi = test::value(makeStateFromVector(vec, *dd));

  EXPECT_TRUE(psi.isTerminal());
  ASSERT_TRUE(dd->getRootSet<vNode>().contains(psi));
  dd->garbageCollect(true);
  EXPECT_TRUE(psi.w.approximatelyEquals(dd->cn.lookup(alpha)));
  EXPECT_NO_THROW(test::value(dd->decRef(psi)));
  EXPECT_TRUE(dd->getRootSet<vNode>().empty());
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

  const std::vector<BasisStates> state{
      BasisStates::plus,
      BasisStates::minus,
      BasisStates::right,
      BasisStates::left,
  };

  auto dd = test::value(Package::create(nq));
  auto const ref = test::value(makeBasisState(nq, state, *dd));
  auto const psi = test::value(makeStateFromVector(vec, *dd));

  EXPECT_EQ(psi, ref);

  test::value(dd->decRef(ref));
  test::value(dd->decRef(psi));
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(StateGenerationTest, MakeZeroInvalidArguments) {

  // Test: Misconfigured package (# of qubits).

  constexpr std::size_t nq = 2;

  auto dd = test::value(Package::create(nq));
  EXPECT_EQ(test::errorKind(makeZeroState(nq + 1, *dd)),
            Error::Kind::InvalidArgument);
}

TEST(StateGenerationTest, MakeBasisInvalidArguments) {

  // Test: Misconfigured package (# of qubits).
  // Test: Invalid size for `state` vector.

  constexpr std::size_t nq = 2;

  auto dd = test::value(Package::create(nq));
  const std::vector<BasisStates> state{BasisStates::one};

  EXPECT_EQ(test::errorKind(makeBasisState(nq + 1, state, *dd)),
            Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(makeBasisState(nq, state, *dd)),
            Error::Kind::InvalidArgument);
}

TEST(StateGenerationTest, MakeGHZInvalidArguments) {

  // Test: Misconfigured package (# of qubits).

  constexpr std::size_t nq = 2;

  auto dd = test::value(Package::create(nq));
  EXPECT_EQ(test::errorKind(makeGHZState(nq + 1, *dd)),
            Error::Kind::InvalidArgument);
}

TEST(StateGenerationTest, MakeWInvalidArguments) {

  // Test: Misconfigured package (# of qubits).

  constexpr std::size_t nq = 2;

  auto dd = test::value(Package::create(nq));
  EXPECT_EQ(test::errorKind(makeWState(nq + 1, *dd)),
            Error::Kind::InvalidArgument);

  const auto tol = dd::RealNumber::eps;
  dd::ComplexNumbers::setTolerance(1);
  EXPECT_EQ(test::errorKind(makeWState(nq, *dd)), Error::Kind::InvalidArgument);
  dd::ComplexNumbers::setTolerance(tol); // Reset tolerance.
}

TEST(StateGenerationTest, FromVectorInvalidArguments) {

  // Test: Misconfigured package (# of qubits).
  // Test: Invalid length of state vector.

  constexpr std::size_t nq = 2;

  auto dd = test::value(Package::create(nq));
  EXPECT_EQ(test::errorKind(makeStateFromVector(CVec(5), *dd)),
            Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(makeStateFromVector(CVec(3), *dd)),
            Error::Kind::InvalidArgument);
}

TEST(StateGenerationTest, VectorConstructionChecksCapacity) {
  auto emptyOwner = test::value(Package::create(0));
  auto& empty = *emptyOwner;
  EXPECT_EQ(test::errorKind(makeStateFromVector(CVec(2), empty)),
            Error::Kind::InvalidArgument);
  auto oneQubitOwner = test::value(Package::create(1));
  auto& oneQubit = *oneQubitOwner;
  EXPECT_EQ(test::errorKind(makeStateFromVector(CVec(4), oneQubit)),
            Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(makeStateFromVector(CVec(8), oneQubit)),
            Error::Kind::InvalidArgument);
  bool read = false;
  const auto entry = [&read](size_t) {
    read = true;
    return std::complex<fp>{};
  };
  EXPECT_EQ(test::errorKind(makeStateFromVector(4, entry, oneQubit)),
            Error::Kind::InvalidArgument);
  EXPECT_FALSE(read);
  const auto state = test::value(
      makeStateFromVector(CVec{{0.5, 0.25}, {-0.5, 0.75}}, oneQubit));
  expectStateVectorNear(state.getVector(), {{0.5, 0.25}, {-0.5, 0.75}});
  EXPECT_NO_THROW(test::value(oneQubit.decRef(state)));
}

TEST(StateGenerationTest, StateIntervalsRejectOverflow) {
  auto packageOwner = test::value(Package::create(2));
  auto& package = *packageOwner;
  const auto maximum = std::numeric_limits<size_t>::max();
  EXPECT_EQ(test::errorKind(makeZeroState(maximum, package)),
            Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(makeZeroState(1, package, maximum)),
            Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(
                makeBasisState(1, std::vector<bool>{false}, package, maximum)),
            Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(makeBasisState(2, std::vector<BasisStates>(2),
                                           package, maximum)),
            Error::Kind::InvalidArgument);
  EXPECT_EQ(test::errorKind(
                makeBasisState(2, std::vector<BasisStates>(2), package, 1)),
            Error::Kind::InvalidArgument);
  const auto state =
      test::value(makeBasisState(1, std::vector<bool>{true}, package, 1));
  ASSERT_FALSE(state.isTerminal());
  EXPECT_EQ(state.p->v, 1);
  EXPECT_TRUE(state.p->e[0].isZeroTerminal());
  EXPECT_TRUE(state.p->e[1].isOneTerminal());
  EXPECT_NO_THROW(test::value(package.decRef(state)));
}

TEST(StateGenerationTest, BasisConstructionUsesRequestedPrefix) {
  auto packageOwner = test::value(Package::create(4));
  auto& package = *packageOwner;
  const std::vector<bool> bits{true, false, true, true, false};
  const std::vector<BasisStates> basis{
      BasisStates::one, BasisStates::zero, BasisStates::one,
      BasisStates::one, BasisStates::zero,
  };
  for (const size_t width : {0U, 1U, 4U}) {
    const auto binary = test::value(makeBasisState(width, bits, package));
    const auto product = test::value(makeBasisState(width, basis, package));
    const auto zero = test::value(makeZeroState(width, package));
    EXPECT_EQ(binary, product);
    EXPECT_EQ(test::value(binary.getValueByIndex(13U & ((1U << width) - 1U))),
              1.);
    EXPECT_EQ(test::value(zero.getValueByIndex(0)), 1.);
    test::value(package.decRef(binary));
    test::value(package.decRef(product));
    test::value(package.decRef(zero));
  }
  EXPECT_EQ(
      test::errorKind(makeBasisState(2, std::vector<bool>{true}, package)),
      Error::Kind::InvalidArgument);
}
