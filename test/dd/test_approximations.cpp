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
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>

using namespace dd;

namespace {
void vecNear(CVec a, CVec b, double delta = 1e-6) {
  for (std::size_t i = 0; i < b.size(); ++i) {
    EXPECT_NEAR(a[i].real(), b[i].real(), delta);
    EXPECT_NEAR(b[i].imag(), b[i].imag(), delta);
  }
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

  // |state⟩ = 0.866|0⟩ + 0.5|1⟩
  //
  // Eliminate nothing (fidelity = 1).
  //     → |approx⟩ = |state⟩

  auto state = simulate(qc, dd.makeZeroState(nq), dd);
  auto approx = approximate(state, fidelity, dd);

  const CVec expected{{0}, {1}};

  EXPECT_EQ(approx.getVector(), expected);
  EXPECT_EQ(approx.size(), 2);
}

TEST(ApproximationTest, OneQubitApproximation) {
  constexpr std::size_t nq = 1;
  constexpr double fidelity = 1 - 0.25;

  Package dd(nq);

  qc::QuantumComputation qc(nq);
  qc.ry(qc::PI / 3, 0);

  // |state⟩ = 0.866|0⟩ + 0.5|1⟩
  //
  // Eliminate |1⟩ with contribution 0.25
  //     → |approx⟩ = |0⟩

  auto state = simulate(qc, dd.makeZeroState(nq), dd);
  auto approx = approximate(state, fidelity, dd);

  const CVec expected{{1}, {0}};

  EXPECT_EQ(approx.getVector(), expected);
  EXPECT_EQ(approx.size(), 2);
  EXPECT_NEAR(dd.fidelity(state, approx), 0.75, 1e-3);

  dd.decRef(approx);
  dd.garbageCollect(true);

  EXPECT_EQ(dd.vUniqueTable.getNumEntries(), 0); // correct ref counts.
}

TEST(ApproximationTest, TwoQubitApproximation) {
  constexpr std::size_t nq = 2;
  constexpr double fidelity = 1 - 0.2;

  Package dd(nq);

  qc::QuantumComputation qc(nq);
  qc.h(0);
  qc.cry(qc::PI / 3, 0, 1);

  // |state⟩ = 0.707|00⟩ + 0.612|01⟩ + 0.354|11⟩
  //
  // Eliminate |11⟩ with contribution 0.125
  //     → |approx⟩ = 0.756|00⟩ + 0.654|01⟩

  auto state = simulate(qc, dd.makeZeroState(nq), dd);
  auto approx = approximate(state, fidelity, dd);

  const CVec expected{{0.755929}, {0.654654}, {0}, {0}};

  vecNear(approx.getVector(), expected);
  EXPECT_EQ(approx.size(), 3);
  EXPECT_NEAR(dd.fidelity(state, approx), 0.875, 1e-3);
}

TEST(ApproximationTest, TwoQubitCorrectlyRebuilt) {
  constexpr std::size_t nq = 2;
  constexpr double fidelity = 1 - 0.1;

  Package dd(nq);

  qc::QuantumComputation qc(nq);
  qc.h(0);
  qc.h(1);
  qc.ry(qc::PI / 3, 0);

  qc::QuantumComputation qcRef(nq);
  qcRef.h(0);
  qcRef.x(1);
  qcRef.cx(0, 1);
  qcRef.cx(1, 0);

  // |state⟩ = 0.183|00⟩ + 0.683|01⟩ + 0.183|10⟩ + 0.683|11⟩
  //
  // Eliminate |00⟩ and |10⟩ with contributions ~ 0.0335.
  //     → |approx⟩ = (1/sqrt(2))(|01⟩ + |11⟩)
  //
  // |ref⟩ = (1/sqrt(2))(|01⟩ + |11⟩)

  auto state = simulate(qc, dd.makeZeroState(nq), dd);
  auto approx = approximate(state, fidelity, dd);
  auto ref = simulate(qcRef, dd.makeZeroState(nq), dd);

  const CVec expected{{0}, {1 / std::sqrt(2)}, {0}, {1 / std::sqrt(2)}};

  vecNear(approx.getVector(), expected);
  EXPECT_EQ(approx.size(), 3);
  EXPECT_NEAR(dd.fidelity(state, approx), 0.933, 1e-3);
  EXPECT_EQ(ref, approx); // points to same node and same edge weight
}

TEST(ApproximationTest, ThreeQubitApproximation) {
  constexpr std::size_t nq = 3;
  constexpr double fidelity = 0.98;

  Package dd(nq);

  qc::QuantumComputation qc(nq);
  qc.h(0);
  qc.cry(qc::PI / 8, 0, 1);
  qc.h(1);
  qc.cry(qc::PI / 8, 1, 2);

  // |state⟩ =   0.5|000⟩ + 0.588|001⟩ +  0.49|010⟩
  //         + 0.385|011⟩ + 0.098|110⟩ + 0.077|111⟩
  //
  // Eliminate parent of |110⟩ and |111> with contribution ~ 0.016.
  //     → |approx⟩ = 0.504|000⟩ + 0.593|001⟩ +  0.494|010⟩ + 0.388|011⟩

  auto state = simulate(qc, dd.makeZeroState(nq), dd);
  auto approx = approximate(state, fidelity, dd);

  const CVec expected{{0.503892}, {0.592515}, {0.49421}, {0.388298},
                      {0},        {0},        {0},       {0}};

  vecNear(approx.getVector(), expected);
  EXPECT_EQ(approx.size(), 5);
  EXPECT_NEAR(dd.fidelity(state, approx), 0.984, 1e-3);
}
