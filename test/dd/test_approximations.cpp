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
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

using namespace dd;

namespace {
std::complex<fp> getNorm(std::vector<std::complex<fp>> vec) {
  return std::inner_product(vec.begin(), vec.end(), vec.begin(),
                            std::complex<fp>());
}
}; // namespace

///-----------------------------------------------------------------------------
///                      \n simulate with approximation \n
///-----------------------------------------------------------------------------

TEST(ApproximationTest, OneQubitKeepAll) {
  constexpr std::size_t nq = 1;
  constexpr double fidelity = 1;

  Package dd(nq);

  qc::QuantumComputation qc(nq);
  qc.x(0);

  // |state⟩ = |1⟩
  auto state = simulate(qc, dd.makeZeroState(nq), dd);
  // fidelity is 1 → keep all nodes: |1⟩
  approximate(state, fidelity, dd);

  CVec expected{{0}, {1}}; // expected state vector for |1⟩: [0, 1]
  EXPECT_EQ(state.getVector(), expected);
  EXPECT_EQ(state.size(), 2); // no nodes deleted: root node + terminal.
}

TEST(ApproximationTest, OneQubitApproximation) {
  constexpr std::size_t nq = 1;
  constexpr double fidelity = 1 - 0.25;

  Package dd(nq);

  qc::QuantumComputation qc(nq);
  qc.ry(qc::PI / 3, 0);

  // |state⟩ = 0.866|0⟩ + 0.5|1⟩
  auto state = simulate(qc, dd.makeZeroState(nq), dd);

  // eliminate |1⟩ with contrib 0.25 → |0⟩
  approximate(state, fidelity, dd);

  CVec expected{{1}, {0}}; // expected state vector for |0⟩: [1, 0]
  EXPECT_EQ(state.getVector(), expected);
  EXPECT_EQ(state.size(), 2); // no nodes deleted: root node + terminal.
}

TEST(ApproximationTest, RemoveOneBottom) {
  constexpr std::size_t nq = 2;
  constexpr double fidelity = 0.98;

  Package dd(nq);

  qc::QuantumComputation qc(nq); // first qubit with prob < 2%.
  qc.h(0);
  qc.cry(qc::PI / 8, 0, 1);

  auto state = simulate(qc, dd.makeZeroState(nq), dd);
  approximate(state, fidelity, dd);

  auto norm = getNorm(state.getVector());

  // has correct number of nodes.
  EXPECT_EQ(state.size(), 3);
  // correct edge is deleted.
  EXPECT_EQ(state.p->e[1], vEdge::zero());
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

  auto state = simulate(qc, dd.makeZeroState(nq), dd);
  approximate(state, fidelity, dd);

  auto norm = getNorm(state.getVector());

  // has correct number of nodes.
  EXPECT_EQ(state.size(), 5);
  // correct edge is deleted.
  EXPECT_EQ(state.p->e[1], vEdge::zero());
  // norm must be one.
  EXPECT_NEAR(norm.real(), 1., 1e-6);
  EXPECT_NEAR(norm.imag(), 0., 1e-6);
}
