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
}

TEST(CliffordBlocks, largerGatethenBlock) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.cx(0, 1);
  qc.x(1);

  QuantumComputation qc2(2);
  qc2.h(0);
  qc2.cx(0, 1);
  qc2.x(1);

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 3);
  EXPECT_TRUE(qc == qc2);
}

TEST(CliffordBlocks, CliffordBlockDepth) {
  QuantumComputation qc(1);
  qc.sx(0);
  qc.h(0);
  qc.x(0);

  QuantumComputation qc2(1);
  QuantumComputation op(1);
  op.sx(0);
  op.h(0);
  op.x(0);
  qc2.emplace_back(op.asCompoundOperation());

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 1);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.front()).size(), 3);
  EXPECT_TRUE(qc == qc2);
}

TEST(CliffordBlocks, CliffordBlockWidth) {
  QuantumComputation qc(2);
  qc.sx(0);
  qc.h(1);

  QuantumComputation qc2(2);
  QuantumComputation op(2);
  op.h(1);
  op.sx(0);
  qc2.emplace_back(op.asCompoundOperation());

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 1);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.front()).size(), 2);
  EXPECT_TRUE(qc == qc2);
}

TEST(CliffordBlocks, keepOrder) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.cx(0, 1);
  qc.sxdg(1);
  qc.z(0);
  qc.y(1);

  QuantumComputation qc2(2);
  QuantumComputation op(2);
  op.h(0);
  op.cx(0, 1);
  op.z(0);
  op.sxdg(1);
  op.y(1);
  qc2.emplace_back(op.asCompoundOperation());

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 1);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.front()).size(), 5);
  EXPECT_TRUE(qc == qc2);
}

TEST(CliffordBlocks, twoCliffordBlocks) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.sxdg(1);
  qc.z(0);
  qc.y(1);

  QuantumComputation qc2(2);
  QuantumComputation op(2);
  QuantumComputation op2(2);
  op.h(0);
  op2.sxdg(1);
  op.z(0);
  op2.y(1);
  qc2.emplace_back(op2.asCompoundOperation());
  qc2.emplace_back(op.asCompoundOperation());

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 2);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_TRUE(qc.back()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.front()).size(), 2);
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.back()).size(), 2);
  EXPECT_TRUE(qc == qc2);
}

TEST(CliffordBlocks, nonCliffordSingleQubit) {
  QuantumComputation qc(1);
  qc.i(0);
  qc.t(0);
  qc.x(0);

  QuantumComputation qc2(1);
  qc2.t(0);
  qc2.x(0);

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 2);
  EXPECT_TRUE(qc == qc2);
}

TEST(CliffordBlocks, SingleNonClifford3Qubit) {
  QuantumComputation qc(3);
  qc.h(0);
  qc.s(1);
  qc.cx(0, 1);
  qc.cx(1, 2);
  qc.rx(0.1, 0);
  qc.cx(0, 1);

  QuantumComputation qc2(3);
  QuantumComputation op(3);
  op.s(1);
  op.h(0);
  op.cx(0, 1);
  op.cx(1, 2);
  qc2.emplace_back(op.asCompoundOperation());
  qc2.rx(0.1, 0);
  qc2.cx(0, 1);

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 3);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 3);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.front()).size(), 4);
  EXPECT_TRUE(qc.back()->isStandardOperation());
  EXPECT_TRUE(qc == qc2);
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

  QuantumComputation qc2(3);
  QuantumComputation op1(3);
  QuantumComputation op2(3);
  op1.s(1);
  op1.h(0);
  op1.cx(0, 1);
  qc2.emplace_back(op1.asCompoundOperation());
  qc2.rx(0.1, 0);
  op2.cx(1, 2);
  op2.y(1);
  qc2.emplace_back(op2.asCompoundOperation());
  qc2.x(0);

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 4);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.front()).size(), 3);
  EXPECT_TRUE(qc.back()->isStandardOperation());
  EXPECT_TRUE(qc == qc2);
}

TEST(CliffordBlocks, shiftedNonClifford) {
  QuantumComputation qc(2);
  qc.cx(0, 1);
  qc.sxdg(1);
  qc.t(0);
  qc.x(0);
  qc.t(1);
  qc.x(1);

  QuantumComputation qc2(2);
  QuantumComputation op(2);
  QuantumComputation op2(2);
  op.cx(0, 1);
  op.sxdg(1);
  qc2.emplace_back(op.asCompoundOperation());
  qc2.t(0);
  qc2.t(1);
  op2.x(0);
  op2.x(1);
  qc2.emplace_back(op2.asCompoundOperation());

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 3);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 4);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.front()).size(), 2);
  EXPECT_TRUE(qc.back()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.back()).size(), 2);
  EXPECT_TRUE(qc == qc2);
}

TEST(CliffordBlocks, nonCliffordBeginning) {
  QuantumComputation qc(2);
  qc.t(0);
  qc.t(1);
  qc.ecr(0, 1);
  qc.x(0);

  QuantumComputation qc2(2);
  qc2.t(1);
  qc2.t(0);
  QuantumComputation op(2);
  op.ecr(0, 1);
  op.x(0);
  qc2.emplace_back(op.asCompoundOperation());

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 3);
  EXPECT_TRUE(qc.front()->isStandardOperation());
  EXPECT_TRUE(qc.back()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.back()).size(), 2);
  EXPECT_TRUE(qc == qc2);
}

TEST(CliffordBlocks, threeQubitnonClifford) {
  QuantumComputation qc(3);
  qc.h(0);
  qc.h(1);
  qc.h(2);
  qc.mcx({0, 1}, 2);
  qc.dcx(0, 1);
  qc.dcx(1, 2);

  QuantumComputation qc2(3);
  QuantumComputation op(3);
  QuantumComputation op2(3);
  op.h(2);
  op.h(1);
  op.h(0);
  qc2.emplace_back(op.asCompoundOperation());
  qc2.mcx({0, 1}, 2);
  op2.dcx(0, 1);
  op2.dcx(1, 2);
  qc2.emplace_back(op2.asCompoundOperation());

  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 3);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 3);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_TRUE(qc.back()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.front()).size(), 3);
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.back()).size(), 2);
  EXPECT_TRUE(qc == qc2);
}

TEST(CliffordBlocks, barrierNotinBlock) {
  QuantumComputation qc(1);
  qc.h(0);
  qc.barrier(0);
  qc.h(0);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 1);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 3);
  EXPECT_TRUE(qc.back()->isStandardOperation());
}

} // namespace qc
