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
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"

#include <gtest/gtest.h>
#include <iostream>

namespace qc {
TEST(ElidePermutations, emptyCircuit) {
  QuantumComputation qc(1);
  qc::CircuitOptimizer::elidePermutations(qc);
  EXPECT_EQ(qc.size(), 0);
}

TEST(ElidePermutations, simpleSwap) {
  QuantumComputation qc(2);
  qc.swap(0, 1);
  qc.h(1);

  std::cout << qc << "\n";
  qc::CircuitOptimizer::elidePermutations(qc);
  std::cout << qc << "\n";

  EXPECT_EQ(qc.size(), 1);
  EXPECT_TRUE(qc.front()->isStandardOperation());
  auto reference = StandardOperation(0, H);
  EXPECT_EQ(*qc.front(), reference);

  EXPECT_EQ(qc.outputPermutation[0], 1);
  EXPECT_EQ(qc.outputPermutation[1], 0);
}

TEST(ElidePermutations, simpleInitialLayout) {
  QuantumComputation qc(1);
  qc.initialLayout = {};
  qc.initialLayout[2] = 0;
  qc.outputPermutation = {};
  qc.outputPermutation[2] = 0;
  qc.h(2);

  std::cout << qc << "\n";
  qc::CircuitOptimizer::elidePermutations(qc);
  std::cout << qc << "\n";

  EXPECT_EQ(qc.size(), 1);
  EXPECT_TRUE(qc.front()->isStandardOperation());
  auto reference = StandardOperation(0, H);
  EXPECT_EQ(*qc.front(), reference);
  EXPECT_EQ(qc.initialLayout[0], 0);
  EXPECT_EQ(qc.outputPermutation[0], 0);
}

TEST(ElidePermutations, applyPermutationCompound) {
  QuantumComputation qc(2);
  qc.cx(0, 1);
  auto op = qc.asCompoundOperation();
  op->addControl(2);
  Permutation perm{};
  perm[0] = 2;
  perm[1] = 1;
  perm[2] = 0;
  op->apply(perm);
  EXPECT_EQ(op->size(), 1);
  EXPECT_TRUE(op->isControlled());
  EXPECT_EQ(op->getControls().size(), 1);
  EXPECT_EQ(op->getControls().begin()->qubit, 0);
  EXPECT_EQ(op->getOps().front()->getTargets().front(), 1);
  EXPECT_EQ(op->getOps().front()->getControls().size(), 2);
  EXPECT_EQ(op->getOps().front()->getControls(), Controls({0, 2}));
}

TEST(ElidePermutations, compoundOperation) {
  QuantumComputation qc(2);
  QuantumComputation op(2);
  op.cx(0, 1);
  op.swap(0, 1);
  op.cx(0, 1);
  qc.emplace_back(op.asOperation());
  qc.cx(1, 0);

  std::cout << qc << "\n";
  qc::CircuitOptimizer::elidePermutations(qc);
  std::cout << qc << "\n";

  EXPECT_EQ(qc.size(), 2);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  auto& compound = dynamic_cast<CompoundOperation&>(*qc.front());
  EXPECT_EQ(compound.size(), 2);
  auto reference = StandardOperation(0, 1, X);
  auto reference2 = StandardOperation(1, 0, X);
  EXPECT_EQ(*compound.getOps().front(), reference);
  EXPECT_EQ(*compound.getOps().back(), reference2);
  EXPECT_EQ(*qc.back(), reference);
  EXPECT_EQ(qc.outputPermutation[0], 1);
  EXPECT_EQ(qc.outputPermutation[1], 0);
}

TEST(ElidePermutations, compoundOperation2) {
  QuantumComputation qc(2);
  QuantumComputation op(2);
  op.swap(0, 1);
  op.cx(0, 1);
  qc.emplace_back(op.asOperation());
  qc.cx(0, 1);

  std::cout << qc << "\n";
  qc::CircuitOptimizer::elidePermutations(qc);
  std::cout << qc << "\n";

  EXPECT_EQ(qc.size(), 2);
  EXPECT_TRUE(qc.front()->isStandardOperation());
  auto reference = StandardOperation(1, 0, X);
  EXPECT_EQ(*qc.front(), reference);
  EXPECT_TRUE(qc.back()->isStandardOperation());
  EXPECT_EQ(*qc.back(), reference);
  EXPECT_EQ(qc.outputPermutation[0], 1);
  EXPECT_EQ(qc.outputPermutation[1], 0);
}

TEST(ElidePermutations, compoundOperation3) {
  QuantumComputation qc(2);
  QuantumComputation op(2);
  op.swap(0, 1);
  qc.emplace_back(op.asCompoundOperation());
  qc.cx(0, 1);

  std::cout << qc << "\n";
  qc::CircuitOptimizer::elidePermutations(qc);
  std::cout << qc << "\n";

  EXPECT_EQ(qc.size(), 1);
  EXPECT_TRUE(qc.front()->isStandardOperation());
  auto reference = StandardOperation(1, 0, X);
  EXPECT_EQ(*qc.front(), reference);
  EXPECT_EQ(qc.outputPermutation[0], 1);
  EXPECT_EQ(qc.outputPermutation[1], 0);
}

TEST(ElidePermutations, compoundOperation4) {
  QuantumComputation qc(3);
  QuantumComputation op(2);
  qc.swap(0, 2);
  op.cx(0, 1);
  op.h(0);
  qc.emplace_back(op.asOperation());
  qc.back()->addControl(2);

  std::cout << qc << "\n";
  qc::CircuitOptimizer::elidePermutations(qc);
  std::cout << qc << "\n";

  EXPECT_EQ(qc.size(), 1);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_TRUE(qc.front()->isControlled());
  EXPECT_EQ(qc.front()->getControls().size(), 1);
  EXPECT_EQ(qc.front()->getControls().begin()->qubit, 0);
  EXPECT_EQ(qc.outputPermutation[0], 2);
  EXPECT_EQ(qc.outputPermutation[1], 1);
  EXPECT_EQ(qc.outputPermutation[2], 0);
}

TEST(ElidePermutations, nonUnitaryOperation) {
  QuantumComputation qc(2, 2);
  qc.swap(0, 1);
  qc.measure(1, 0);
  qc.outputPermutation[0] = 1;
  qc.outputPermutation[1] = 0;

  std::cout << qc << "\n";
  qc::CircuitOptimizer::elidePermutations(qc);
  std::cout << qc << "\n";

  EXPECT_EQ(qc.size(), 1);
  EXPECT_TRUE(qc.front()->isNonUnitaryOperation());
  auto reference = NonUnitaryOperation(0, 0);
  EXPECT_EQ(*qc.front(), reference);
  EXPECT_EQ(qc.outputPermutation[0], 0);
  EXPECT_EQ(qc.outputPermutation[1], 1);
}
} // namespace qc
