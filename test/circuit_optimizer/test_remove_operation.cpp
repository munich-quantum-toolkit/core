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
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/OpType.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>

namespace qc {
TEST(RemoveOperation, removeIdentities) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits);
  qc.i(0);
  qc.i(0);
  qc.x(0);
  qc.i(0);
  qc.i(0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeIdentities(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
}

TEST(RemoveOperation, removeSingleQubitGates) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits);
  qc.x(0);
  qc.x(0);
  qc.y(0);
  qc.i(0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeOperation(qc, {X, Y}, 1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
}

TEST(RemoveOperation, removeMultiQubitGates) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.x(0);
  qc.cx(0, 1);
  qc.cy(0, 1);
  qc.cz(0, 1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeOperation(qc, {X, Y}, 2);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 2);
}

TEST(RemoveOperation, removeMoves) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.x(0);
  qc.move(0, 1);
  qc.cy(1, 1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeOperation(qc, {Move}, 0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 2);
}

TEST(RemoveOperation, removeGateInCompoundOperation) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits);
  QuantumComputation compound(nqubits);
  compound.x(0);
  compound.y(0);
  compound.z(0);
  qc.emplace_back(compound.asOperation());
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeOperation(qc, {Y}, 1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_EQ(qc.front()->getType(), Compound);
  const auto& compoundOp = dynamic_cast<const CompoundOperation&>(*qc.front());
  EXPECT_EQ(compoundOp.size(), 2);
}
} // namespace qc
