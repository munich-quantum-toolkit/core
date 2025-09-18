/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/RandomCliffordCircuit.hpp"
#include "circuit_optimizer/CircuitOptimizer.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace qc {
TEST(FlattenOperations, FlattenRandomClifford) {
  auto rcs = createRandomCliffordCircuit(2U, 3U, 0U);
  std::cout << rcs << "\n";
  const auto nops = rcs.getNindividualOps();

  CircuitOptimizer::flattenOperations(rcs);
  std::cout << rcs << "\n";

  for (const auto& op : rcs) {
    EXPECT_FALSE(op->isCompoundOperation());
  }
  EXPECT_EQ(nops, rcs.getNindividualOps());
}

TEST(FlattenOperations, FlattenRecursive) {
  const std::size_t nqubits = 1U;

  // create a nested compound operation
  QuantumComputation op(nqubits);
  op.x(0);
  op.z(0);
  QuantumComputation op2(nqubits);
  op2.emplace_back(op.asCompoundOperation());
  QuantumComputation qc(nqubits);
  qc.emplace_back(op2.asCompoundOperation());
  std::cout << qc << "\n";

  CircuitOptimizer::flattenOperations(qc);
  std::cout << qc << "\n";

  for (const auto& g : qc) {
    EXPECT_FALSE(g->isCompoundOperation());
  }

  ASSERT_EQ(qc.getNops(), 2U);
  auto& gate = qc.at(0);
  EXPECT_EQ(gate->getType(), X);
  EXPECT_EQ(gate->getTargets().at(0), 0U);
  EXPECT_TRUE(gate->getControls().empty());
  auto& gate2 = qc.at(1);
  EXPECT_EQ(gate2->getType(), Z);
  EXPECT_EQ(gate2->getTargets().at(0), 0U);
  EXPECT_TRUE(gate2->getControls().empty());
}

TEST(FlattenOperations, FlattenCustomOnly) {
  const std::size_t nqubits = 1U;

  // create a nested compound operation
  QuantumComputation op(nqubits);
  op.x(0);
  op.z(0);
  QuantumComputation op2(nqubits);
  op2.emplace_back(op.asCompoundOperation());
  QuantumComputation qc(nqubits);
  qc.emplace_back(op2.asCompoundOperation());
  std::cout << qc << "\n";

  CircuitOptimizer::flattenOperations(qc, true);
  std::cout << qc << "\n";

  ASSERT_EQ(qc.getNops(), 1U);
  auto& gate = qc.at(0);
  EXPECT_EQ(gate->getType(), Compound);

  std::vector<std::unique_ptr<Operation>> opsCompound;
  opsCompound.push_back(std::make_unique<StandardOperation>(0, X));
  opsCompound.push_back(std::make_unique<StandardOperation>(0, Z));
  QuantumComputation qc2(nqubits);
  qc2.emplace_back<CompoundOperation>(std::move(opsCompound), true);
  std::cout << qc2 << "\n";

  CircuitOptimizer::flattenOperations(qc2, true);
  std::cout << qc2 << "\n";

  for (const auto& g : qc2) {
    EXPECT_FALSE(g->isCompoundOperation());
  }

  ASSERT_EQ(qc2.getNops(), 2U);
  auto& gate3 = qc2.at(0);
  EXPECT_EQ(gate3->getType(), X);
  EXPECT_EQ(gate3->getTargets().at(0), 0U);
  EXPECT_TRUE(gate3->getControls().empty());
  auto& gate4 = qc2.at(1);
  EXPECT_EQ(gate4->getType(), Z);
  EXPECT_EQ(gate4->getTargets().at(0), 0U);
  EXPECT_TRUE(gate4->getControls().empty());
}
} // namespace qc
