/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "circuit_optimizer/CircuitOptimizer.hpp"
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>

namespace qc {
TEST(ReplaceMCXwithMCZ, replaceCXwithCZ) {
  qc::QuantumComputation qc(2U);
  qc.cx(0, 1);
  CircuitOptimizer::replaceMCXWithMCZ(qc);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.getNops(), 3U);
  EXPECT_EQ(qc.at(0)->getType(), qc::H);
  EXPECT_EQ(qc.at(0)->getTargets()[0], 1U);
  EXPECT_EQ(qc.at(1)->getType(), qc::Z);
  EXPECT_EQ(qc.at(1)->getTargets()[0], 1U);
  EXPECT_EQ(*qc.at(1)->getControls().begin(), 0U);
  EXPECT_EQ(qc.at(2)->getType(), qc::H);
  EXPECT_EQ(qc.at(2)->getTargets()[0], 1U);
}

TEST(ReplaceMCXwithMCZ, replaceCCXwithCCZ) {
  std::size_t const nqubits = 3U;
  qc::QuantumComputation qc(nqubits);
  Controls const controls = {0, 1};
  Qubit const target = 2U;
  qc.mcx(controls, target);
  CircuitOptimizer::replaceMCXWithMCZ(qc);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.getNops(), 3U);
  EXPECT_EQ(qc.at(0)->getType(), qc::H);
  EXPECT_EQ(qc.at(0)->getTargets()[0], target);
  EXPECT_EQ(qc.at(1)->getType(), qc::Z);
  EXPECT_EQ(qc.at(1)->getTargets()[0], target);
  EXPECT_EQ(qc.at(1)->getControls(), controls);
  EXPECT_EQ(qc.at(2)->getType(), qc::H);
  EXPECT_EQ(qc.at(2)->getTargets()[0], target);
}

TEST(ReplaceMCXwithMCZ, replaceCXwithCZinCompoundOperation) {
  std::size_t const nqubits = 2U;
  qc::QuantumComputation op(nqubits);
  op.cx(0, 1);

  qc::QuantumComputation qc(nqubits);
  qc.emplace_back(op.asCompoundOperation());

  CircuitOptimizer::replaceMCXWithMCZ(qc);
  std::cout << qc << "\n";

  CircuitOptimizer::flattenOperations(qc);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.getNops(), 3U);
  EXPECT_EQ(qc.at(0)->getType(), qc::H);
  EXPECT_EQ(qc.at(0)->getTargets()[0], 1U);
  EXPECT_EQ(qc.at(1)->getType(), qc::Z);
  EXPECT_EQ(qc.at(1)->getTargets()[0], 1U);
  EXPECT_EQ(*qc.at(1)->getControls().begin(), 0U);
  EXPECT_EQ(qc.at(2)->getType(), qc::H);
  EXPECT_EQ(qc.at(2)->getTargets()[0], 1U);
}

TEST(ReplaceMCXwithMCZ, testToffoliSequenceSimplification) {
  std::size_t const nqubits = 3U;
  qc::QuantumComputation qc(nqubits);
  Controls const controls = {0, 1};
  Qubit const target = 2U;
  qc.cx(0, target);
  qc.mcx(controls, target);
  CircuitOptimizer::replaceMCXWithMCZ(qc);
  CircuitOptimizer::singleQubitGateFusion(qc);
  CircuitOptimizer::flattenOperations(qc);
  std::cout << qc << "\n";

  qc::QuantumComputation reference(nqubits);
  reference.h(target);
  reference.cz(0, target);
  reference.mcz(controls, target);
  reference.h(target);

  for (std::size_t i = 0; i < reference.getNops(); ++i) {
    EXPECT_EQ(qc.at(i)->getType(), reference.at(i)->getType());
    EXPECT_EQ(qc.at(i)->getTargets(), reference.at(i)->getTargets());
    EXPECT_EQ(qc.at(i)->getControls(), reference.at(i)->getControls());
  }
}
} // namespace qc
