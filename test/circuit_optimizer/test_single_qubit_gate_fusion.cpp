/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "circuit_optimizer/CircuitOptimizer.hpp"
#include "ir/QuantumComputation.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>

namespace qc {

TEST(SingleQubitGateFusion, CollapseCompoundOperationToStandard) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits);
  qc.x(0);
  qc.i(0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_TRUE(qc.begin()->get()->isStandardOperation());
}

TEST(SingleQubitGateFusion, eliminateCompoundOperation) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits);
  qc.i(0);
  qc.i(0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 0);
  EXPECT_TRUE(qc.empty());
}

TEST(SingleQubitGateFusion, eliminateInverseInCompoundOperation) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits);
  qc.s(0);
  qc.sdg(0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 0);
  EXPECT_TRUE(qc.empty());
}

TEST(SingleQubitGateFusion, unknownInverseInCompoundOperation) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits);
  qc.p(1., 0);
  qc.p(-1., 0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
}

TEST(SingleQubitGateFusion, repeatedCancellationInSingleQubitGateFusion) {
  const std::size_t nqubits = 1U;
  QuantumComputation qc(nqubits);
  qc.x(0);
  qc.h(0); // causes the creation of a CompoundOperation
  qc.h(0); // cancels a gate in the compound operation
  qc.x(0); // cancels the first gate, making the CompoundOperation empty
  qc.z(0); // adds another gate to the CompoundOperation
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1U);
}

TEST(SingleQubitGateFusion, emptyCompoundGatesRemovedInSingleQubitGateFusion) {
  const std::size_t nqubits = 1U;
  QuantumComputation qc(nqubits);
  qc.x(0);
  qc.h(0); // causes the creation of a CompoundOperation
  qc.h(0); // cancels a gate in the compound operation
  qc.x(0); // cancels the first gate, making the CompoundOperation empty
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 0U);
}

TEST(SingleQubitGateFusion, SingleQubitGateCount) {
  QuantumComputation qc(2U, 2U);
  qc.x(0);
  qc.h(0);
  qc.cx(1, 0);
  qc.z(0);
  qc.measure(0, 0);

  EXPECT_EQ(qc.getNops(), 5U);
  EXPECT_EQ(qc.getNindividualOps(), 5U);
  EXPECT_EQ(qc.getNsingleQubitOps(), 3U);

  CircuitOptimizer::singleQubitGateFusion(qc);

  EXPECT_EQ(qc.getNops(), 4U);
  EXPECT_EQ(qc.getNindividualOps(), 5U);
  EXPECT_EQ(qc.getNsingleQubitOps(), 3U);
}

} // namespace qc
