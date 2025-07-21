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

TEST(CliffordBlocks, nonCliffordOnAll) {
  QuantumComputation qc(2);
  qc.sx(0);
  qc.cx(0, 1);
  qc.sxdg(1);
  qc.t(0);
  qc.x(0);
  qc.t(1);
  qc.x(1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 3);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 4);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.front()).size(), 3);
}

TEST(CliffordBlocks, nonCliffordSingleQubit) {
  QuantumComputation qc(1);
  qc.sdg(0);
  qc.h(0);
  qc.t(0);
  qc.x(0);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 3);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.front()).size(), 2);
}

TEST(CliffordBlocks, collectTwoQubitCliffordGates) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.s(1);
  qc.cx(0, 1);
  qc.rx(0.1, 0);
  qc.x(0);
  qc.y(1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 3);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_TRUE(qc.back()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.front()).size(), 3);
}

TEST(CliffordBlocks, TwoQubitnonClifford) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.s(1);
  qc.rxx(0.1, 0, 1);
  qc.i(0);
  qc.y(1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 3);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_TRUE(qc.back()->isStandardOperation());
}

TEST(CliffordBlocks, mergeBlocksnonClifford) {
  QuantumComputation qc(3);
  qc.h(0);
  qc.cx(0, 1);
  qc.cx(1, 2);
  qc.t(0);
  qc.t(1);
  qc.cx(0, 1);
  qc.cx(1, 2);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 3);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 4);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.front()).size(), 3);
  EXPECT_TRUE(qc.back()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.back()).size(), 2);
}

TEST(CliffordBlocks, nonCliffordBeginning) {
  QuantumComputation qc(2);
  qc.t(0);
  qc.t(1);
  qc.ecr(0, 1);
  qc.x(0);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 2);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 3);
  EXPECT_TRUE(qc.front()->isStandardOperation());
  EXPECT_TRUE(qc.back()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.back()).size(), 2);
}

TEST(CliffordBlocks, threeQubitnonClifford) {
  QuantumComputation qc(3);
  qc.h(0);
  qc.h(1);
  qc.h(2);
  qc.mcx({0, 1}, 2);
  qc.mcz({0, 2}, 1);
  qc.dcx(0, 1);
  qc.dcx(1, 2);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::collectCliffordBlocks(qc, 3);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.size(), 4);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_TRUE(qc.back()->isCompoundOperation());
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.front()).size(), 3);
  EXPECT_EQ(dynamic_cast<qc::CompoundOperation&>(*qc.back()).size(), 2);
}

} // namespace qc
