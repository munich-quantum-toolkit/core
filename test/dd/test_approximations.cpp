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
///                      \n compute node contributions \n
///-----------------------------------------------------------------------------

TEST(ApproximationTest, NodeContributionsSingleEdge) {
  constexpr std::size_t nq = 1;
  Package dd(nq);

  qc::QuantumComputation qc(nq); // i|1>
  qc.x(0);
  qc.s(0);

  VectorDD root = simulate(qc, dd.makeZeroState(nq), dd);
  NodeContributions contributions(root);

  EXPECT_NEAR(contributions[root.p], 1., 1e-6);
}

TEST(ApproximationTest, NodeContributionsGHZ) {
  constexpr std::size_t nq = 2;
  Package dd(nq);

  qc::QuantumComputation qc(nq); // 1/sqrt(2) * (|00> + |11>)
  qc.h(1);
  qc.cx(1, 0);

  VectorDD root = simulate(qc, dd.makeZeroState(nq), dd);
  NodeContributions contributions(root);

  EXPECT_NEAR(contributions[root.p], 1., 1e-6);
  EXPECT_NEAR(contributions[root.p->e[0].p], .5, 1e-6);
  EXPECT_NEAR(contributions[root.p->e[1].p], .5, 1e-6);
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

  EXPECT_NEAR(contributions[root.p], 1., 1e-6);
  EXPECT_NEAR(contributions[root.p->e[0].p], 1., 1e-6);
}

TEST(ApproximationTest, NodeContributions2Percent) {
  constexpr std::size_t nq = 2;
  Package dd(nq);

  qc::QuantumComputation qc(nq); // first qubit with prob < 2%.
  qc.h(0);
  qc.cry(qc::PI / 8, 0, 1);

  VectorDD root = simulate(qc, dd.makeZeroState(nq), dd);
  NodeContributions contributions(root);

  EXPECT_NEAR(contributions[root.p], 1., 1e-6);
  EXPECT_NEAR(contributions[root.p->e[0].p], .98096988, 1e-6);
  EXPECT_NEAR(contributions[root.p->e[1].p], .01903012, 1e-6);
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

  EXPECT_NEAR(contributions[root.p], 1., 1e-6);
  EXPECT_NEAR(contributions[root.p->e[1].p], 1., 1e-6);
  EXPECT_NEAR(contributions[root.p->e[1].p->e[0].p], .5, 1e-6);
  EXPECT_NEAR(contributions[root.p->e[1].p->e[1].p], .5, 1e-6);
}

TEST(ApproximationTest, NodeContributionsTwoCRY) {
  constexpr std::size_t nq = 3;
  Package dd(nq);

  qc::QuantumComputation qc(nq);
  qc.h(0);
  qc.cry(qc::PI / 8, 0, 1);
  qc.h(1);
  qc.cry(qc::PI / 8, 1, 2);

  VectorDD root = simulate(qc, dd.makeZeroState(nq), dd);
  NodeContributions contributions(root);

  EXPECT_NEAR(contributions[root.p], 1., 1e-6);
  EXPECT_NEAR(contributions[root.p->e[0].p], .984611, 1e-6);
  EXPECT_NEAR(contributions[root.p->e[1].p], .0153889, 1e-6);
  EXPECT_NEAR(contributions[root.p->e[0].p->e[0].p], .595671, 1e-6);
  EXPECT_NEAR(contributions[root.p->e[0].p->e[1].p], .404329, 1e-6);
  EXPECT_NEAR(contributions[root.p->e[1].p->e[1].p], .404329, 1e-6);
}

///-----------------------------------------------------------------------------
///                      \n simulate with approximation \n
///-----------------------------------------------------------------------------

TEST(ApproximationTest, FidelityDrivenKeepAll) {
  constexpr std::size_t nq = 2;
  Package dd(nq);

  qc::QuantumComputation qc(nq);
  qc.x(0);

  VectorDD root = simulate(qc, dd.makeZeroState(nq), dd);

  constexpr Approximation<FidelityDriven> approx(1.);
  applyApproximation(root, approx, dd);

  EXPECT_EQ(root.size(), 3);
}

TEST(ApproximationTest, MemoryDriven2Percent) {
  constexpr std::size_t nq = 2;
  Package dd(nq);

  qc::QuantumComputation qc(nq); // first qubit with prob < 2%.
  qc.h(0);
  qc.cry(qc::PI / 8, 0, 1);

  VectorDD root = simulate(qc, dd.makeZeroState(nq), dd);

  constexpr Approximation<MemoryDriven> approx(3, 0.98);
  applyApproximation(root, approx, dd);

  EXPECT_EQ(root.size(), 3);
  EXPECT_EQ(root.p->e[1], vEdge::zero());

  NodeContributions contributions(root);
  EXPECT_NEAR(contributions[root.p], 1., 1e-6);
  EXPECT_NEAR(contributions[root.p->e[0].p], 1., 1e-6);
}

TEST(ApproximationTest, MemoryDrivenTwoCRY) {
  constexpr std::size_t nq = 3;
  Package dd(nq);

  qc::QuantumComputation qc(nq);
  qc.h(0);
  qc.cry(qc::PI / 8, 0, 1);
  qc.h(1);
  qc.cry(qc::PI / 8, 1, 2);

  VectorDD root = simulate(qc, dd.makeZeroState(nq), dd);

  constexpr Approximation<MemoryDriven> approx(5, 0.98);
  applyApproximation(root, approx, dd);

  EXPECT_EQ(root.size(), 5);
  EXPECT_EQ(root.p->e[1], vEdge::zero());

  NodeContributions contributions(root);
  EXPECT_NEAR(contributions[root.p], 1., 1e-6);
}

TEST(ApproximationTest, MemoryDrivenKeepAll) {
  constexpr std::size_t nq = 2;
  Package dd(nq);

  qc::QuantumComputation qc(nq); // first qubit with prob < 2%.
  qc.h(0);
  qc.cry(qc::PI / 8, 0, 1);

  VectorDD root = simulate(qc, dd.makeZeroState(nq), dd);

  constexpr Approximation<MemoryDriven> approx(4, 0.98);
  applyApproximation(root, approx, dd);

  EXPECT_EQ(root.size(), 4);
}
