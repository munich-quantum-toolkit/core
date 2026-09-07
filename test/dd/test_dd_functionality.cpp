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
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "dd/StateGeneration.hpp"

#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <numbers>

namespace dd {

TEST(DDGateConstruction, AppliesGlobalPhase) {
  Package package(1);
  auto state = makeZeroState(1, package);

  const auto phased = applyGlobalPhase(state, std::numbers::pi / 2., package);
  EXPECT_EQ(state, phased);
  const auto vector = phased.getVector();

  ASSERT_EQ(vector.size(), 2);
  EXPECT_NEAR(vector[0].real(), 0., RealNumber::eps);
  EXPECT_NEAR(vector[0].imag(), 1., RealNumber::eps);
  EXPECT_EQ(vector[1], std::complex<fp>{});
}

TEST(DDGateConstruction, VectorKroneckerWithTerminal) {
  constexpr std::size_t nq = 1;
  constexpr auto root = vEdge::one();
  Package package(nq);

  const auto zeroState = makeZeroState(nq, package);
  const auto extendedRoot = package.kronecker(zeroState, root, 0);
  EXPECT_EQ(zeroState, extendedRoot);

  package.decRef(zeroState);
  package.garbageCollect(true);

  const auto [vector, matrix, reals] = package.computeActiveCounts();
  EXPECT_EQ(vector, 0);
  EXPECT_EQ(matrix, 0);
  EXPECT_EQ(reals, 0);
}

} // namespace dd
