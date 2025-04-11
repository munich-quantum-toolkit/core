/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Approximation.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "ir/QuantumComputation.hpp"

#include <gtest/gtest.h>

using namespace dd;

///-----------------------------------------------------------------------------
///                      \n simulate with approximation \n
///-----------------------------------------------------------------------------

TEST(ApproximationTest, KeepAll) {
  constexpr std::size_t nq = 2;
  Package dd(nq);

  qc::QuantumComputation qc(nq);
  qc.x(0);

  auto root = simulate(qc, dd.makeZeroState(nq), dd);
  auto approx = approximate(root, 0.5, dd);
  EXPECT_EQ(root, approx);
  EXPECT_EQ(dd.fidelity(root, approx), 1);
}

TEST(ApproximationTest, RemoveOneBottom) {
  constexpr std::size_t nq = 2;
  Package dd(nq);

  qc::QuantumComputation qc(nq); // first qubit with prob < 2%.
  qc.h(0);
  qc.cry(qc::PI / 8, 0, 1);

  auto root = simulate(qc, dd.makeZeroState(nq), dd);
  auto approx = approximate(root, 0.98, dd);

  EXPECT_EQ(root.size(), 4);
  EXPECT_EQ(approx.size(), 3);
  EXPECT_NE(root, approx);
  EXPECT_EQ(approx.p->e[1], vEdge::zero());
  EXPECT_NEAR(dd.fidelity(root, approx), 0.98, 1e-2);
}

TEST(ApproximationTest, RemoveOneMiddle) {
  constexpr std::size_t nq = 3;
  Package dd(nq);

  qc::QuantumComputation qc(nq);
  qc.h(0);
  qc.cry(qc::PI / 8, 0, 1);
  qc.h(1);
  qc.cry(qc::PI / 8, 1, 2);

  auto root = simulate(qc, dd.makeZeroState(nq), dd);
  auto approx = approximate(root, 0.98, dd);

  EXPECT_EQ(root.size(), 6);
  EXPECT_EQ(approx.size(), 5);
  EXPECT_NE(root, approx);
  EXPECT_EQ(approx.p->e[1], vEdge::zero());
  EXPECT_NEAR(dd.fidelity(root, approx), 0.98, 1e-2);
}
