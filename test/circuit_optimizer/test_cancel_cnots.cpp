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

#include <gtest/gtest.h>

namespace qc {
TEST(CancelCNOTs, CNOTCancellation1) {
  QuantumComputation qc(2);
  qc.cx(1, 0);
  qc.cx(1, 0);

  CircuitOptimizer::cancelCNOTs(qc);
  EXPECT_TRUE(qc.empty());
}

TEST(CancelCNOTs, CNOTCancellation2) {
  QuantumComputation qc(2);
  qc.swap(0, 1);
  qc.swap(1, 0);

  CircuitOptimizer::cancelCNOTs(qc);
  EXPECT_TRUE(qc.empty());
}

TEST(CancelCNOTs, CNOTCancellation3) {
  QuantumComputation qc(2);
  qc.swap(0, 1);
  qc.cx(1, 0);

  CircuitOptimizer::cancelCNOTs(qc);
  EXPECT_TRUE(qc.size() == 2U);
  const auto& firstOperation = qc.front();
  EXPECT_EQ(firstOperation->getType(), qc::X);
  EXPECT_EQ(firstOperation->getTargets().front(), 0U);
  EXPECT_EQ(firstOperation->getControls().begin()->qubit, 1U);

  const auto& secondOperation = qc.back();
  EXPECT_EQ(secondOperation->getType(), qc::X);
  EXPECT_EQ(secondOperation->getTargets().front(), 1U);
  EXPECT_EQ(secondOperation->getControls().begin()->qubit, 0U);
}

TEST(CancelCNOTs, CNOTCancellation4) {
  QuantumComputation qc(2);
  qc.cx(1, 0);
  qc.swap(0, 1);

  CircuitOptimizer::cancelCNOTs(qc);
  EXPECT_TRUE(qc.size() == 2U);
  const auto& firstOperation = qc.front();
  EXPECT_EQ(firstOperation->getType(), qc::X);
  EXPECT_EQ(firstOperation->getTargets().front(), 1U);
  EXPECT_EQ(firstOperation->getControls().begin()->qubit, 0U);

  const auto& secondOperation = qc.back();
  EXPECT_EQ(secondOperation->getType(), qc::X);
  EXPECT_EQ(secondOperation->getTargets().front(), 0U);
  EXPECT_EQ(secondOperation->getControls().begin()->qubit, 1U);
}

TEST(CancelCNOTs, CNOTCancellation5) {
  QuantumComputation qc(2);
  qc.cx(1, 0);
  qc.cx(0, 1);
  qc.cx(1, 0);

  CircuitOptimizer::cancelCNOTs(qc);
  EXPECT_TRUE(qc.size() == 1U);
  const auto& firstOperation = qc.front();
  EXPECT_EQ(firstOperation->getType(), qc::SWAP);
  EXPECT_EQ(firstOperation->getTargets().front(), 0U);
  EXPECT_EQ(firstOperation->getTargets().back(), 1U);
}
} // namespace qc
