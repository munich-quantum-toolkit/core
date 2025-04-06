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
#include "ir/operations/OpType.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>

namespace qc {
TEST(RemoveDiagonalGateBeforeMeasure, removeDiagonalSingleQubitBeforeMeasure) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits, nqubits);
  qc.z(0);
  qc.measure(0, 0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_EQ(qc.begin()->get()->getType(), qc::Measure);
}

TEST(RemoveDiagonalGateBeforeMeasure, removeDiagonalCompoundOpBeforeMeasure) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits, nqubits);
  qc.z(0);
  qc.t(0);
  qc.measure(0, 0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_EQ(qc.begin()->get()->getType(), qc::Measure);
}

TEST(RemoveDiagonalGateBeforeMeasure, removeDiagonalTwoQubitGateBeforeMeasure) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  qc.cz(0, 1);
  qc.measure({0, 1}, {0, 1});
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_EQ(qc.begin()->get()->getType(), qc::Measure);
}

TEST(RemoveDiagonalGateBeforeMeasure, leaveGateBeforeMeasure) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  qc.cz(0, 1);
  qc.x(0);
  qc.measure({0, 1}, {0, 1});
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 3);
}

TEST(RemoveDiagonalGateBeforeMeasure, removeComplexGateBeforeMeasure) {
  const std::size_t nqubits = 4;
  QuantumComputation qc(nqubits, nqubits);
  qc.cz(0, 1);
  qc.x(0);
  qc.cz(1, 2);
  qc.cz(0, 1);
  qc.z(0);
  qc.cz(1, 2);
  qc.x(3);
  qc.t(3);
  qc.mcz({0, 1, 2}, 3);
  qc.measure({0, 1, 2, 3}, {0, 1, 2, 3});
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 4);
}

TEST(RemoveDiagonalGateBeforeMeasure, removeSimpleCompoundOpBeforeMeasure) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits, nqubits);
  qc.x(0);
  qc.t(0);
  qc.measure(0, 0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 2);
}

TEST(RemoveDiagonalGateBeforeMeasure, removePartOfCompoundOpBeforeMeasure) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits, nqubits);
  qc.t(0);
  qc.x(0);
  qc.t(0);
  qc.measure(0, 0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 2);
}
} // namespace qc
