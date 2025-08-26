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

TEST(CliffordBlocks, emptyCircuit) {
  QuantumComputation qc(1);
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 1);
  EXPECT_EQ(qc.getNindividualOps(), 0);
}

TEST(CliffordBlocks, singleGate) {
  QuantumComputation qc(1);
  qc.sx(0);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 1);
  EXPECT_FALSE(qc.front()->isCompoundOperation());
}

TEST(CliffordBlocks, largerGatethenBlock) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.cx(0, 1);
  qc.x(1);

  QuantumComputation expectedQc(2);
  expectedQc.h(0);
  expectedQc.cx(0, 1);
  expectedQc.x(1);

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_TRUE(qc == expectedQc);
}

TEST(CliffordBlocks, CliffordBlockDepth) {
  QuantumComputation qc(1);
  qc.sx(0);
  qc.h(0);
  qc.x(0);

  QuantumComputation expectedQc(1);
  QuantumComputation op1(1);
  op1.sx(0);
  op1.h(0);
  op1.x(0);
  expectedQc.emplace_back(op1.asCompoundOperation());

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_TRUE(qc == expectedQc);
}

TEST(CliffordBlocks, CliffordBlockWidth) {
  QuantumComputation qc(2);
  qc.sx(0);
  qc.h(1);

  QuantumComputation expectedQc(2);
  QuantumComputation op1(2);
  op1.h(1);
  op1.sx(0);
  expectedQc.emplace_back(op1.asCompoundOperation());

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_TRUE(qc == expectedQc);
}

TEST(CliffordBlocks, keepOrder) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.cx(0, 1);
  qc.sxdg(1);
  qc.z(0);
  qc.y(1);

  QuantumComputation expectedQc(2);
  QuantumComputation op1(2);
  op1.h(0);
  op1.cx(0, 1);
  op1.z(0);
  op1.sxdg(1);
  op1.y(1);
  expectedQc.emplace_back(op1.asCompoundOperation());

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_TRUE(qc == expectedQc);
}

TEST(CliffordBlocks, twoCliffordBlocks) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.sxdg(1);
  qc.z(0);
  qc.y(1);

  QuantumComputation expectedQc(2);
  QuantumComputation op1(2);
  QuantumComputation op2(2);
  op1.sxdg(1);
  op1.y(1);
  expectedQc.emplace_back(op1.asCompoundOperation());

  op2.h(0);
  op2.z(0);
  expectedQc.emplace_back(op2.asCompoundOperation());

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_TRUE(qc == expectedQc);
}

TEST(CliffordBlocks, nonCliffordSingleQubit) {
  QuantumComputation qc(1);
  qc.i(0);
  qc.t(0);
  qc.x(0);

  QuantumComputation expectedQc(1);
  expectedQc.t(0);
  expectedQc.x(0);

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_TRUE(qc == expectedQc);
}

TEST(CliffordBlocks, SingleNonClifford3Qubit) {
  QuantumComputation qc(3);
  qc.h(0);
  qc.s(1);
  qc.cx(0, 1);
  qc.cx(1, 2);
  qc.rx(0.1, 0);
  qc.cx(0, 1);

  QuantumComputation expectedQc(3);
  QuantumComputation op1(3);
  op1.s(1);
  op1.h(0);
  op1.cx(0, 1);
  op1.cx(1, 2);

  expectedQc.emplace_back(op1.asCompoundOperation());
  expectedQc.rx(0.1, 0);
  expectedQc.cx(0, 1);

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 3);
  std::cout << qc << "\n";
  EXPECT_TRUE(qc == expectedQc);
}

TEST(CliffordBlocks, TwoCliffordBlocks3Qubit) {
  QuantumComputation qc(3);
  qc.h(0);
  qc.s(1);
  qc.cx(0, 1);
  qc.cx(1, 2);
  qc.rx(0.1, 0);
  qc.x(0);
  qc.y(1);

  QuantumComputation expectedQc(3);
  QuantumComputation op1(3);
  QuantumComputation op2(3);
  op1.s(1);
  op1.h(0);
  op1.cx(0, 1);
  expectedQc.emplace_back(op1.asCompoundOperation());
  expectedQc.rx(0.1, 0);

  op2.cx(1, 2);
  op2.y(1);
  expectedQc.emplace_back(op2.asCompoundOperation());
  expectedQc.x(0);

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_TRUE(qc == expectedQc);
}

TEST(CliffordBlocks, shiftedNonClifford) {
  QuantumComputation qc(2);
  qc.cx(0, 1);
  qc.sxdg(1);
  qc.t(0);
  qc.x(0);
  qc.t(1);
  qc.x(1);

  QuantumComputation expectedQc(2);
  QuantumComputation op1(2);
  QuantumComputation op2(2);
  op1.cx(0, 1);
  op1.sxdg(1);

  expectedQc.emplace_back(op1.asCompoundOperation());
  expectedQc.t(0);
  expectedQc.t(1);

  op2.x(0);
  op2.x(1);
  expectedQc.emplace_back(op2.asCompoundOperation());

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 3);
  std::cout << qc << "\n";
  EXPECT_TRUE(qc == expectedQc);
}

TEST(CliffordBlocks, nonCliffordBeginning) {
  QuantumComputation qc(2);
  qc.t(0);
  qc.t(1);
  qc.ecr(0, 1);
  qc.x(0);

  QuantumComputation expectedQc(2);
  QuantumComputation op1(2);
  expectedQc.t(1);
  expectedQc.t(0);

  op1.ecr(0, 1);
  op1.x(0);
  expectedQc.emplace_back(op1.asCompoundOperation());

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_TRUE(qc == expectedQc);
}

TEST(CliffordBlocks, threeQubitnonClifford) {
  QuantumComputation qc(3);
  qc.h(0);
  qc.h(1);
  qc.h(2);
  qc.mcx({0, 1}, 2);
  qc.dcx(0, 1);
  qc.dcx(1, 2);

  QuantumComputation expectedQc(3);
  QuantumComputation op1(3);
  QuantumComputation op2(3);
  op1.h(2);
  op1.h(1);
  op1.h(0);

  expectedQc.emplace_back(op1.asCompoundOperation());
  expectedQc.mcx({0, 1}, 2);
  
  op2.dcx(0, 1);
  op2.dcx(1, 2);
  expectedQc.emplace_back(op2.asCompoundOperation());

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 3);
  std::cout << qc << "\n";
  EXPECT_TRUE(qc == expectedQc);
}

TEST(CliffordBlocks, handleCompoundOperation) {
  QuantumComputation qc(1);
  QuantumComputation op1(1);
  op1.x(0);
  op1.z(0);

  qc.h(0);
  qc.emplace_back(op1.asCompoundOperation());

  QuantumComputation expectedQc(1);
  QuantumComputation op2(1);
  QuantumComputation op3(1);
  op2.x(0);
  op2.z(0);

  op3.h(0);
  op3.emplace_back(op2.asCompoundOperation());
  expectedQc.emplace_back(op3.asCompoundOperation());

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_TRUE(expectedQc == qc);
}

TEST(CliffordBlocks, handleCompoundOperation2) {
  QuantumComputation qc(1);
  QuantumComputation op1(1);
  qc.h(0);
  op1.t(0);
  op1.z(0);
  op1.x(0);
  qc.emplace_back(op1.asCompoundOperation());

  QuantumComputation expectedQc(1);
  QuantumComputation op2(1);
  expectedQc.h(0);
  op2.t(0);
  op2.z(0);
  op2.x(0);
  expectedQc.emplace_back(op2.asCompoundOperation());

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_TRUE(qc == expectedQc);
}

TEST(CliffordBlocks, barrierNotinBlock) {
  QuantumComputation qc(1);
  qc.h(0);
  qc.barrier(0);
  qc.h(0);

  QuantumComputation expectedQc(1);
  expectedQc.h(0);
  expectedQc.barrier(0);
  expectedQc.h(0);

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_TRUE(qc == expectedQc);
}

} // namespace qc
