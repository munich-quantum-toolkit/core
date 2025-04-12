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
#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "ir/QuantumComputation.hpp"

#include <complex>
#include <gtest/gtest.h>
#include <numeric>

using namespace dd;

std::complex<fp> getNorm(std::vector<std::complex<fp>> vec) {
  return std::inner_product(vec.begin(), vec.end(), vec.begin(),
                            std::complex<fp>());
}

///-----------------------------------------------------------------------------
///                      \n simulate with approximation \n
///-----------------------------------------------------------------------------

TEST(ApproximationTest, KeepAll) {
  constexpr std::size_t nq = 2;
  constexpr double fidelity = 1;
  Package dd(nq);

  qc::QuantumComputation qc(nq);
  qc.x(0);

  auto root = simulate(qc, dd.makeZeroState(nq), dd);
  auto approx = approximate(root, fidelity, dd);

  auto norm = getNorm(approx.getVector());

  // no nodes deleted. must be the same.
  EXPECT_EQ(root, approx);
  // final fidelity is correct.
  EXPECT_EQ(dd.fidelity(root, approx), 1);
  // norm must be one.
  EXPECT_NEAR(norm.real(), 1., 1e-6);
  EXPECT_NEAR(norm.imag(), 0., 1e-6);
}

TEST(ApproximationTest, RemoveOneBottom) {
  constexpr std::size_t nq = 2;
  constexpr double fidelity = 0.98;

  Package dd(nq);

  qc::QuantumComputation qc(nq); // first qubit with prob < 2%.
  qc.h(0);
  qc.cry(qc::PI / 8, 0, 1);

  auto root = simulate(qc, dd.makeZeroState(nq), dd);
  auto approx = approximate(root, fidelity, dd);

  auto norm = getNorm(approx.getVector());

  // has correct number of nodes.
  EXPECT_EQ(root.size(), 4);
  EXPECT_EQ(approx.size(), 3);
  // can't be the same.
  EXPECT_NE(root, approx);
  // correct edge is deleted.
  EXPECT_EQ(approx.p->e[1], vEdge::zero());
  // final fidelity is correct.
  EXPECT_NEAR(dd.fidelity(root, approx), fidelity, 1e-2);
  // norm must be one.
  EXPECT_NEAR(norm.real(), 1., 1e-6);
  EXPECT_NEAR(norm.imag(), 0., 1e-6);
}

TEST(ApproximationTest, RemoveOneMiddle) {
  constexpr std::size_t nq = 3;
  constexpr double fidelity = 0.98;

  Package dd(nq);

  qc::QuantumComputation qc(nq);
  qc.h(0);
  qc.cry(qc::PI / 8, 0, 1);
  qc.h(1);
  qc.cry(qc::PI / 8, 1, 2);

  auto root = simulate(qc, dd.makeZeroState(nq), dd);
  auto approx = approximate(root, fidelity, dd);

  auto norm = getNorm(approx.getVector());

  // has correct number of nodes.
  EXPECT_EQ(root.size(), 6);
  EXPECT_EQ(approx.size(), 5);
  // can't be the same.
  EXPECT_NE(root, approx);
  // correct edge is deleted.
  EXPECT_EQ(approx.p->e[1], vEdge::zero());
  // final fidelity is correct.
  EXPECT_NEAR(dd.fidelity(root, approx), fidelity, 1e-2);
  // norm must be one.
  EXPECT_NEAR(norm.real(), 1., 1e-6);
  EXPECT_NEAR(norm.imag(), 0., 1e-6);
}
