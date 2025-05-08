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

namespace qc {
TEST(SwapReconstruction, fuseCxToSwap) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.cx(0, 1);
  qc.cx(1, 0);
  qc.cx(0, 1);
  CircuitOptimizer::swapReconstruction(qc);
  const auto& op = qc.front();
  EXPECT_TRUE(op->isStandardOperation());
  EXPECT_EQ(op->getType(), SWAP);
  EXPECT_EQ(op->getTargets().at(0), 0);
  EXPECT_EQ(op->getTargets().at(1), 1);
}

TEST(SwapReconstruction, replaceCxToSwapAtEnd) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.cx(0, 1);
  qc.cx(1, 0);
  CircuitOptimizer::swapReconstruction(qc);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isStandardOperation());
  EXPECT_EQ(op->getType(), SWAP);
  EXPECT_EQ(op->getTargets().at(0), 0);
  EXPECT_EQ(op->getTargets().at(1), 1);

  ++it;
  const auto& op2 = *it;
  EXPECT_TRUE(op2->isStandardOperation());
  EXPECT_EQ(op2->getType(), X);
  EXPECT_EQ(op2->getControls().begin()->qubit, 0);
  EXPECT_EQ(op2->getTargets().at(0), 1);
}

TEST(SwapReconstruction, replaceCxToSwap) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.cx(0, 1);
  qc.cx(1, 0);
  qc.h(0);
  CircuitOptimizer::swapReconstruction(qc);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isStandardOperation());
  EXPECT_EQ(op->getType(), SWAP);
  EXPECT_EQ(op->getTargets().at(0), 0);
  EXPECT_EQ(op->getTargets().at(1), 1);
  ++it;
  const auto& op2 = *it;
  EXPECT_TRUE(op2->isStandardOperation());
  EXPECT_EQ(op2->getType(), X);
  EXPECT_EQ(op2->getControls().begin()->qubit, 0);
  EXPECT_EQ(op2->getTargets().at(0), 1);
}
} // namespace qc
