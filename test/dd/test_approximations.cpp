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

TEST(ApproximationTest, NodeContributionsSingleEdge) {
  constexpr std::size_t nq = 1;
  Package dd(nq);

  qc::QuantumComputation qc(nq); // i|1>
  qc.x(0);
  qc.s(0);

  VectorDD root = simulate(qc, dd.makeZeroState(nq), dd);
  NodeContributions contributions(root);

  EXPECT_NEAR(contributions[root.p], 1., 1e-8);
}

TEST(ApproximationTest, NodeContributionsGHZ) {
  constexpr std::size_t nq = 2;
  Package dd(nq);

  qc::QuantumComputation qc(nq); // 1/sqrt(2) * (|00> + |11>)
  qc.h(1);
  qc.cx(1, 0);

  VectorDD root = simulate(qc, dd.makeZeroState(nq), dd);
  NodeContributions contributions(root);

  EXPECT_NEAR(contributions[root.p], 1., 1e-8);
  EXPECT_NEAR(contributions[root.p->e[0].p], .5, 1e-8);
  EXPECT_NEAR(contributions[root.p->e[1].p], .5, 1e-8);
}

TEST(ApproximationTest, NodeContributionsDoubleVisit) {
  constexpr std::size_t nq = 2;
  Package dd(nq);

  qc::QuantumComputation qc(nq); // 1/2 * (|00> - |01> + |10> - |11>)
  qc.h(1);
  qc.x(0);
  qc.h(0);

  VectorDD root = simulate(qc, dd.makeZeroState(nq), dd);
  NodeContributions contributions(root);

  EXPECT_NEAR(contributions[root.p], 1., 1e-8);
  EXPECT_NEAR(contributions[root.p->e[0].p], 1., 1e-8);
}

TEST(ApproximationTest, NodeContributions2Percent) {
  constexpr std::size_t nq = 2;
  Package dd(nq);

  qc::QuantumComputation qc(nq); // first qubit with prob < 2%.
  qc.h(0);
  qc.cry(qc::PI / 8, 0, 1);

  VectorDD root = simulate(qc, dd.makeZeroState(nq), dd);
  NodeContributions contributions(root);

  EXPECT_NEAR(contributions[root.p], 1., 1e-8);
  EXPECT_NEAR(contributions[root.p->e[0].p], .98096988, 1e-8);
  EXPECT_NEAR(contributions[root.p->e[1].p], .01903012, 1e-8);
}

TEST(ApproximationTest, NodeContributionsGrover) {
  constexpr std::size_t nq = 3;
  Package dd(nq);

  // Grover after 1st H in diffusion (Oracle |11>).
  qc::QuantumComputation qc(nq);
  qc.h(0);
  qc.h(1);
  qc.x(2);
  qc.mcz(qc::Controls{0, 1}, 2);
  qc.h(0);
  qc.h(1);

  VectorDD root = simulate(qc, dd.makeZeroState(nq), dd);
  NodeContributions contributions(root);

  EXPECT_NEAR(contributions[root.p], 1., 1e-8);
  EXPECT_NEAR(contributions[root.p->e[1].p], 1., 1e-8);
  EXPECT_NEAR(contributions[root.p->e[1].p->e[0].p], .5, 1e-8);
  EXPECT_NEAR(contributions[root.p->e[1].p->e[1].p], .5, 1e-8);
}

TEST(ApproximationTest, FidelityDrivenF1) {
  constexpr std::size_t nq = 2;

  std::mt19937_64 mt;
  mt.seed(42);

  // Setup package.
  Package dd(nq);

  // Circuit to simulate.
  qc::QuantumComputation qc(nq);
  qc.x(0);

  // Call simulate.
  Approximation<FidelityDriven> approx(1.);
  VectorDD res = simulate(qc, dd.makeZeroState(nq), dd, approx);
  const std::string m = dd.measureAll(res, false, mt, 0.001);

  EXPECT_EQ(res.size(), 3); // number of nodes + terminal.
  EXPECT_EQ(m, "01");       // (kron(I, X))|00> = |01>
}

TEST(ApproximationTest, FidelityDriven2Percent) {
  constexpr std::size_t nq = 2;
  Package dd(nq);

  qc::QuantumComputation qc(nq); // first qubit with prob < 2%.
  qc.h(0);
  qc.cry(qc::PI / 8, 0, 1);

  Approximation<FidelityDriven> approx(.98);
  VectorDD root = simulate(qc, dd.makeZeroState(nq), dd, approx);

  EXPECT_EQ(root.size(), 2);
}
