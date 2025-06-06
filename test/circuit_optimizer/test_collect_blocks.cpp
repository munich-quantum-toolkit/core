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

#include <gtest/gtest.h>
#include <iostream>

namespace qc {
TEST(CollectBlocks, emptyCircuit) {
  QuantumComputation qc(1);
  qc::CircuitOptimizer::collectBlocks(qc, 1);
  EXPECT_EQ(qc.getNindividualOps(), 0);
}

TEST(CollectBlocks, singleGate) {
  QuantumComputation qc(1);
  qc.h(0);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 1);
  EXPECT_TRUE(qc.front()->isStandardOperation());
}

TEST(CollectBlocks, collectMultipleSingleQubitGates) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.h(1);
  qc.x(0);
  qc.x(1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 2);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_TRUE(qc.back()->isCompoundOperation());
}

TEST(CollectBlocks, nonCliffordOnAll) {
  QuantumComputation qc(2);
  qc.sx(0);
  qc.sxdg(1);
  qc.t(0);
  qc.t(1);
  qc.x(0);
  qc.x(1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 2, true);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 4);
}

TEST(CollectBlocks, nonCliffordonTop) {
  QuantumComputation qc(2);
  qc.sdg(0);
  qc.h(1);
  qc.t(0);
  qc.x(0);
  qc.z(1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 2, true);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 3);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
}

TEST(CollectBlocks, collectTwoQubitCliffordGates) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.s(1);
  qc.cx(0, 1);
  qc.rx(0.1, 0);
  qc.x(0);
  qc.y(1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 2, true);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 2);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
}

TEST(CollectBlocks, TwoQubitnonClifford) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.s(1);
  qc.rxx(0.1, 0, 1);
  qc.i(0);
  qc.y(1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 2, true);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 3);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
}

TEST(CollectBlocks, nonCliffordBeginning) {
  QuantumComputation qc(2);
  qc.t(0);
  qc.t(1);
  qc.ecr(0, 1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 2, true);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 3);
}

TEST(CollectBlocks, threeQubitnonClifford) {
  QuantumComputation qc(3);
  qc.mcx({0, 1}, 2);
  qc.mcz({0, 2}, 1);
  qc.dcx(0, 1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 2, true);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 3);
}

TEST(CollectBlocks, mergeBlocks) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.h(1);
  qc.cx(0, 1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 1);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.front()).size(), 3);
}

TEST(CollectBlocks, mergeBlocks2) {
  QuantumComputation qc(2);
  qc.h(1);
  qc.x(1);
  qc.h(1);
  qc.z(0);
  qc.cx(0, 1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 1);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.front()).size(), 5);
}

TEST(CollectBlocks, addToMultiQubitBlock) {
  QuantumComputation qc(2);
  qc.cx(0, 1);
  qc.cz(0, 1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 1);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.front()).size(), 2);
}

TEST(CollectBlocks, gateTooBig) {
  QuantumComputation qc(3);
  qc.h(0);
  qc.h(1);
  qc.mcx({0, 1}, 2);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 2);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_TRUE(qc.back()->isStandardOperation());
}

TEST(CollectBlocks, gateTooBig2) {
  QuantumComputation qc(3);
  qc.h(0);
  qc.h(1);
  qc.mcx({0, 1}, 2);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 3);
  EXPECT_TRUE(qc.front()->isStandardOperation());
  EXPECT_TRUE(qc.back()->isStandardOperation());
}

TEST(CollectBlocks, gateTooBig3) {
  QuantumComputation qc(5);
  qc.cx(0, 1);
  qc.cx(2, 3);
  qc.h(4);
  qc.mcx({0, 1, 2, 3}, 4);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 3);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 3);
  EXPECT_TRUE(qc.back()->isStandardOperation());
}

TEST(CollectBlocks, endingBlocks) {
  QuantumComputation qc(3);
  qc.h(0);
  qc.cx(1, 2);
  qc.cx(0, 1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 2);
  EXPECT_TRUE(qc.front()->isStandardOperation());
  EXPECT_TRUE(qc.back()->isCompoundOperation());
}

TEST(CollectBlocks, endingBlocks2) {
  QuantumComputation qc(4);
  qc.cx(0, 1);
  qc.cx(1, 2);
  qc.mcx({0, 1}, 3);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 3);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 2);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_TRUE(qc.back()->isStandardOperation());
}

TEST(CollectBlocks, interruptBlock) {
  QuantumComputation qc(1);
  qc.h(0);
  qc.reset(0);
  qc.h(0);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 3);
  EXPECT_TRUE(qc.front()->isStandardOperation());
  EXPECT_TRUE(qc.back()->isStandardOperation());
}

TEST(CollectBlocks, unprocessableAtBegin) {
  QuantumComputation qc(1);
  qc.reset(0);
  qc.h(0);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 2);
  EXPECT_TRUE(qc.front()->isNonUnitaryOperation());
  EXPECT_TRUE(qc.back()->isStandardOperation());
}

TEST(CollectBlocks, handleCompoundOperation) {
  QuantumComputation qc(2);
  QuantumComputation op(1);
  op.h(0);
  qc.emplace_back(op.asCompoundOperation());
  qc.x(1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 2);
  EXPECT_TRUE(qc.front()->isStandardOperation());
  EXPECT_TRUE(qc.back()->isStandardOperation());
}

TEST(CollectBlocks, handleCompoundOperation2) {
  QuantumComputation qc(1);
  QuantumComputation op(1);
  op.h(0);
  op.x(0);
  qc.emplace_back(op.asCompoundOperation());
  qc.x(0);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 1);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
}

} // namespace qc
