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

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <random>
#include <ratio>

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

vEdge buildExponentialDD(Qubit v, std::size_t height, Package& dd,
                         std::uniform_real_distribution<float>& dis,
                         std::mt19937& gen) {
  if (v == static_cast<Qubit>(height)) {
    auto e = vEdge::one();
    e.w = dd.cn.lookup(static_cast<fp>(dis(gen)));
    return e;
  }

  const std::array<vEdge, RADIX> edges{
      buildExponentialDD(v + 1, height, dd, dis, gen),
      buildExponentialDD(v + 1, height, dd, dis, gen),
  };

  vEdge eNew = dd.makeDDNode(v, edges);
  eNew.w = dd.cn.lookup(static_cast<fp>(dis(gen)));

  return eNew;
}

}; // namespace

///-----------------------------------------------------------------------------
///                      \n simulate with approximation \n
///-----------------------------------------------------------------------------

TEST(ApproximationTest, OneQubitKeepAllBudgetZero) {

  // Test: If the budget is 0, no approximation will be applied.
  //
  // |state⟩ = |1⟩
  //
  // Eliminate nothing (fidelity = 1).
  //     → |approx⟩ = |state⟩
  //
  //    1│                     1│
  //   ┌─┴─┐                  ┌─┴─┐
  // ┌─│ q0│─┐  -(approx)→  ┌─│ q0│─┐
  // 0 └───┘ │1             0 └───┘ │1
  //         □                      □
  //

  constexpr std::size_t nq = 1;
  constexpr double fidelity = 1;

  auto dd = std::make_unique<dd::Package>(nq);

  qc::QuantumComputation qc(nq);
  qc.x(0);

  auto state = simulate(qc, dd->makeZeroState(nq), *dd);
  auto finalFidelity = approximate(state, fidelity, *dd);

  const CVec expected{{0}, {1}};
  EXPECT_EQ(state.getVector(), expected);
  EXPECT_EQ(state.size(), 2);
  EXPECT_EQ(finalFidelity, 1);
}

TEST(ApproximationTest, OneQubitKeepAllBudgetTooSmall) {

  // Test: If the budget is too small, no approximation will be applied.
  //
  // |state⟩ = |1⟩
  //
  // Eliminate nothing:
  //     → |approx⟩ = |state⟩
  //
  //    1│                     1│
  //   ┌─┴─┐                  ┌─┴─┐
  // ┌─│ q0│─┐  -(approx)→  ┌─│ q0│─┐
  // 0 └───┘ │1             0 └───┘ │1
  //         □                      □
  //

  constexpr std::size_t nq = 1;
  constexpr double fidelity = 0.9;

  auto dd = std::make_unique<dd::Package>(nq);

  qc::QuantumComputation qc(nq);
  qc.x(0);

  auto state = simulate(qc, dd->makeZeroState(nq), *dd);
  auto finalFidelity = approximate(state, fidelity, *dd);

  const CVec expected{{0}, {1}};
  EXPECT_EQ(state.getVector(), expected);
  EXPECT_EQ(state.size(), 2);
  EXPECT_EQ(finalFidelity, 1);
}

TEST(ApproximationTest, TwoQubitRemoveNode) {

  // Test: Remove node (its in-going edge) from decision diagram.
  //
  // |state⟩ = 0.707|00⟩ + 0.612|01⟩ + 0.354|11⟩
  //
  // Eliminate |11⟩ with contribution 0.125
  //     → |approx⟩ = 0.756|00⟩ + 0.654|01⟩
  //
  //          1│                                 1│
  //         ┌─┴─┐                              ┌─┴─┐
  //     ┌───│ q1│───┐                        ┌─│ q1│─┐
  //  .94│   └───┘   │1/(2√2)                1| └───┘ 0
  //   ┌─┴─┐       ┌─┴─┐      -(approx)→    ┌─┴─┐
  // ┌─│ q0│─┐   ┌─│ q0│─┐                ┌─│ q0│─┐
  // | └───┘ |   0 └───┘ |1           .756| └───┘ |.654
  // |.76    |.65        |                □       □
  // □       □           □
  //

  constexpr std::size_t nq = 2;
  constexpr double fidelity = 1 - 0.2;

  auto dd = std::make_unique<dd::Package>(nq);

  qc::QuantumComputation qc(nq);
  qc.h(0);
  qc.cry(qc::PI / 3, 0, 1);

  auto state = simulate(qc, dd->makeZeroState(nq), *dd);
  auto finalFidelity = approximate(state, fidelity, *dd);

  const CVec expected{{0.755929}, {0.654654}, {0}, {0}};

  vecNear(state.getVector(), expected);
  EXPECT_EQ(state.size(), 3);
  EXPECT_NEAR(finalFidelity, 0.875, 1e-3);

  // Test: Correctly increase and decrease ref counts.

  dd->decRef(state);
  dd->garbageCollect(true);

  EXPECT_EQ(dd->vUniqueTable.getNumEntries(), 0);
}

TEST(ApproximationTest, TwoQubitCorrectlyRebuilt) {

  // Test: Compare the approximated source state with an equal constructed
  //       state. The root edge must point to the same node and have the same
  //       edge weight.
  //
  // |state⟩ = i(0.5|00⟩ + 0.866|11⟩)
  //
  // Eliminate |00⟩ with contribution ~ 0.25.
  //     → |approx⟩ = (1/sqrt(2))(|01⟩ + |11⟩)
  //
  // |ref⟩ = |11⟩
  //
  //           i│                           i│
  //          ┌─┴─┐                        ┌─┴─┐
  //      ┌───│ q1│───┐                  ┌─│ q1│─┐
  //  -1/2│   └───┘   │.87               0 └───┘ |1
  //    ┌─┴─┐       ┌─┴─┐     -(approx)→       ┌─┴─┐
  //  ┌─│ q0│─┐   ┌─│ q0│─┐                  ┌─│ q0│─┐
  // 1| └───┘ 0   0 └───┘ |1                 0 └───┘ |1
  //  □                   □                          □

  constexpr std::size_t nq = 2;
  constexpr double fidelity = 1 - 0.25;

  auto dd = std::make_unique<dd::Package>(nq);

  qc::QuantumComputation qc(nq);
  qc.rx(qc::PI, 0);
  qc.ry(2 * qc::PI / 3, 0);
  qc.cx(0, 1);
  qc.x(0);
  qc.x(1);

  qc::QuantumComputation qcRef(nq);
  qcRef.x(0);
  qcRef.s(0);
  qcRef.x(1);

  auto state = simulate(qc, dd->makeZeroState(nq), *dd);
  auto finalFidelity = approximate(state, fidelity, *dd);
  auto ref = simulate(qcRef, dd->makeZeroState(nq), *dd);

  const CVec expected{{0}, {0}, {0}, {0, 1}};
  vecNear(state.getVector(), expected);
  state.printVector();
  EXPECT_EQ(state.size(), 3);
  EXPECT_NEAR(finalFidelity, 0.75, 1e-3);
  EXPECT_EQ(ref, state); // implicit: utilize `==` operator.
}

TEST(ApproximationTest, ThreeQubitRemoveNodeWithChildren) {

  // Test: Remove node that has a subtree attached (i.e. has children).
  //
  // |state⟩ = 0+0.866j|000⟩ + 0-0.096j|100⟩ + 0+0.166j|101⟩ + 0+0.231j|110⟩
  //           + 0-0.4j|111⟩
  //
  // Eliminate parent of |1xx⟩ with contribution ~ 0.25.
  //     → |approx⟩ = i|000⟩
  //
  //               i│                              i│
  //              ┌─┴─┐                           ┌─┴─┐
  //          ┌───│ q2│───┐                     ┌─│ q2│─┐
  //       .87│   └───┘   │-1/2                1| └───┘ 0
  //        ┌─┴─┐       ┌─┴─┐    -(approx)→   ┌─┴─┐
  //      ┌─│ q1│─┐   ┌─│ q1│─┐             ┌─│ q1│─┐
  //      | └───┘ 0   | └───┘ |            1| └───┘ 0
  //      |1      -.38└───┬───┘.92          |
  //    ┌─┴─┐           ┌─┴─┐             ┌─┴─┐
  //  ┌─│ q0│─┐       ┌─│ q0│─┐         ┌─│ q0│─┐
  // 1| └───┘ 0   -1/2| └───┘ |.87     1| └───┘ 0
  //  □               □       □         □
  //

  constexpr std::size_t nq = 3;
  constexpr double fidelity = 1 - 0.25;

  auto dd = std::make_unique<dd::Package>(nq);

  qc::QuantumComputation qc(nq);
  qc.rx(qc::PI, 0);
  qc.ry(2 * qc::PI / 3, 0);
  qc.cx(0, 1);
  qc.cx(1, 2);
  qc.cry(qc::PI / 3, 2, 0);
  qc.cry(qc::PI / 4, 2, 1);

  auto state = simulate(qc, dd->makeZeroState(nq), *dd);
  auto finalFidelity = approximate(state, fidelity, *dd);

  const CVec expected{{0, 1}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};
  vecNear(state.getVector(), expected);
  EXPECT_EQ(state.size(), 4);
  EXPECT_NEAR(finalFidelity, 0.75, 1e-3);
}

TEST(ApproximationTest, ThreeQubitRemoveUnconnected) {

  // Test: Remove multiple nodes.
  //
  // |state⟩ = 0+0.866j|000⟩ + 0-0.096j|100⟩ + 0+0.166j|101⟩ + 0+0.231j|110⟩
  //           + 0-0.4j|111⟩
  //
  // Eliminate parent of |1xx⟩ and |0xx⟩ with contributions ~ 0.25 and ~ 0.75.
  //     → |approx⟩ = 0
  //
  //               i│                              i│
  //              ┌─┴─┐                           ┌─┴─┐
  //          ┌───│ q2│───┐                     ┌─│ q2│─┐
  //       .87│   └───┘   │-1/2                1| └───┘ 0
  //        ┌─┴─┐       ┌─┴─┐    -(approx)→   ┌─┴─┐
  //      ┌─│ q1│─┐   ┌─│ q1│─┐             ┌─│ q1│─┐
  //      | └───┘ 0   | └───┘ |            1| └───┘ 0
  //      |1      -.38└───┬───┘.92          |
  //    ┌─┴─┐           ┌─┴─┐             ┌─┴─┐
  //  ┌─│ q0│─┐       ┌─│ q0│─┐         ┌─│ q0│─┐
  // 1| └───┘ 0   -1/2| └───┘ |.87     1| └───┘ 0
  //  □               □       □         □
  //

  constexpr std::size_t nq = 3;
  constexpr double fidelity = 0; // Eliminate all.

  auto dd = std::make_unique<dd::Package>(nq);

  qc::QuantumComputation qc(nq);
  qc.rx(qc::PI, 0);
  qc.ry(2 * qc::PI / 3, 0);
  qc.cx(0, 1);
  qc.cx(1, 2);
  qc.cry(qc::PI / 3, 2, 0);
  qc.cry(qc::PI / 4, 2, 1);

  auto state = simulate(qc, dd->makeZeroState(nq), *dd);
  auto finalFidelity = approximate(state, fidelity, *dd);

  const CVec expected{{0, 0}};
  vecNear(state.getVector(), expected);
  EXPECT_EQ(state.size(), 1);
  EXPECT_EQ(finalFidelity, 0);
}

TEST(ApproximationTest, Runtime) {
  {
    constexpr std::size_t n = 15;         // Up to 16 qubits.
    constexpr double fidelity = 1 - 0.01; // Budget of .02

    std::array<std::size_t, n> qubits{}; // Qubit counts: [2, 16]
    std::iota(qubits.begin(), qubits.end(), 2);

    // Setup random distribution for edge weights.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0., 1.0);

    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t nq = qubits[i];

      auto dd = std::make_unique<dd::Package>(nq);

      // Build exponentially large DD and increase its reference count.
      auto state = buildExponentialDD(0, nq, *dd, dis, gen);
      dd->incRef(state);

      const std::size_t numNodes =
          state.size() - 1; // Subtract 1 due to terminal.

      const auto t1 = std::chrono::high_resolution_clock::now();
      approximate(state, fidelity, *dd);
      const auto t2 = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double, std::micro> runtime = t2 - t1;

      const double q = runtime.count() /
                       (static_cast<double>(numNodes) * std::log2(numNodes));

      std::cout << nq << " | " << q << '\n';
    }
    EXPECT_FALSE(true);
  }
}
