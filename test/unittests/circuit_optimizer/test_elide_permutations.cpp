#include "CircuitOptimizer.hpp"
#include "QuantumComputation.hpp"

#include "gtest/gtest.h"
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
  auto reference = StandardOperation(0_pc, 1, X);
  auto reference2 = StandardOperation(1_pc, 0, X);
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
  auto reference = StandardOperation(1_pc, 0, X);
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
  auto reference = StandardOperation(1_pc, 0, X);
  EXPECT_EQ(*qc.front(), reference);
  EXPECT_EQ(qc.outputPermutation[0], 1);
  EXPECT_EQ(qc.outputPermutation[1], 0);
}

TEST(ElidePermutations, compoundOperation4) {
  QuantumComputation qc(3);
  QuantumComputation op(2);
  qc.swap(0, 1);
  op.cx(0, 1);
  op.h(0);
  qc.emplace_back(op.asOperation());
  qc.back()->addControl(2);

  std::cout << qc << "\n";
  qc::CircuitOptimizer::elidePermutations(qc);
  std::cout << qc << "\n";

  EXPECT_EQ(qc.size(), 1);
  EXPECT_TRUE(qc.front()->isCompoundOperation());
  EXPECT_EQ(qc.outputPermutation[0], 1);
  EXPECT_EQ(qc.outputPermutation[1], 0);
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
