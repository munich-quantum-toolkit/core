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
TEST(DecomposeSwap, decomposeSWAPsUndirectedArchitecture) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  qc.swap(0, 1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::decomposeSWAP(qc, false);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isStandardOperation());
  EXPECT_EQ(op->getType(), qc::X);
  EXPECT_EQ(op->getControls().begin()->qubit, 0);
  EXPECT_EQ(op->getTargets().at(0), 1);
  ++it;
  const auto& op2 = *it;
  EXPECT_TRUE(op2->isStandardOperation());
  EXPECT_EQ(op2->getType(), qc::X);
  EXPECT_EQ(op2->getControls().begin()->qubit, 1);
  EXPECT_EQ(op2->getTargets().at(0), 0);
  ++it;
  const auto& op3 = *it;
  EXPECT_TRUE(op3->isStandardOperation());
  EXPECT_EQ(op3->getType(), qc::X);
  EXPECT_EQ(op3->getControls().begin()->qubit, 0);
  EXPECT_EQ(op3->getTargets().at(0), 1);
}
TEST(DecomposeSwap, decomposeSWAPsDirectedArchitecture) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.swap(0, 1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::decomposeSWAP(qc, true);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isStandardOperation());
  EXPECT_EQ(op->getType(), qc::X);
  EXPECT_EQ(op->getControls().begin()->qubit, 0);
  EXPECT_EQ(op->getTargets().at(0), 1);

  ++it;
  const auto& op2 = *it;
  EXPECT_TRUE(op2->isStandardOperation());
  EXPECT_EQ(op2->getType(), qc::H);
  EXPECT_EQ(op2->getTargets().at(0), 1);

  ++it;
  const auto& op3 = *it;
  EXPECT_TRUE(op3->isStandardOperation());
  EXPECT_EQ(op3->getType(), qc::H);
  EXPECT_EQ(op3->getTargets().at(0), 0);

  ++it;
  const auto& op4 = *it;
  EXPECT_TRUE(op4->isStandardOperation());
  EXPECT_EQ(op4->getType(), qc::X);
  EXPECT_EQ(op4->getControls().begin()->qubit, 0);
  EXPECT_EQ(op4->getTargets().at(0), 1);

  ++it;
  const auto& op5 = *it;
  EXPECT_TRUE(op5->isStandardOperation());
  EXPECT_EQ(op5->getType(), qc::H);
  EXPECT_EQ(op5->getTargets().at(0), 1);

  ++it;
  const auto& op6 = *it;
  EXPECT_TRUE(op6->isStandardOperation());
  EXPECT_EQ(op6->getType(), qc::H);
  EXPECT_EQ(op6->getTargets().at(0), 0);

  ++it;
  const auto& op7 = *it;
  EXPECT_TRUE(op7->isStandardOperation());
  EXPECT_EQ(op7->getType(), qc::X);
  EXPECT_EQ(op7->getControls().begin()->qubit, 0);
  EXPECT_EQ(op7->getTargets().at(0), 1);
}

TEST(DecomposeSwap, decomposeSWAPsCompound) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  QuantumComputation comp(nqubits);
  comp.swap(0, 1);
  comp.swap(0, 1);
  comp.swap(0, 1);
  qc.emplace_back(comp.asOperation());
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  qc::CircuitOptimizer::decomposeSWAP(qc, false);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isCompoundOperation());

  const auto* cop = dynamic_cast<qc::CompoundOperation*>(op.get());
  ASSERT_NE(cop, nullptr);
  EXPECT_EQ(cop->size(), 9);
}

TEST(DecomposeSwap, decomposeSWAPsCompoundDirected) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  QuantumComputation comp(nqubits);
  comp.swap(0, 1);
  comp.swap(0, 1);
  comp.swap(0, 1);
  qc.emplace_back(comp.asOperation());
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  qc::CircuitOptimizer::decomposeSWAP(qc, true);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isCompoundOperation());

  const auto* cop = dynamic_cast<qc::CompoundOperation*>(op.get());
  ASSERT_NE(cop, nullptr);
  EXPECT_EQ(cop->size(), 21);
}
} // namespace qc
