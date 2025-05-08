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

namespace qc {
TEST(BackpropagateOutputPermutation, FullySpecified) {
  // i *  *      i 1  0
  //   |  |  ->    |  |
  // o 1  0      o 1  0

  auto qc = QuantumComputation(2);
  qc.outputPermutation.clear();
  qc.outputPermutation[0] = 1;
  qc.outputPermutation[1] = 0;
  CircuitOptimizer::backpropagateOutputPermutation(qc);
  EXPECT_EQ(qc.initialLayout, qc.outputPermutation);
}

TEST(BackpropagateOutputPermutation, PartiallySpecifiedQubitAvailable) {
  // i *  *      i 1  0
  //   |  |  ->    |  |
  // o 1  *      o 1  0

  auto qc = QuantumComputation(2);
  qc.outputPermutation.clear();
  qc.outputPermutation[0] = 1;
  CircuitOptimizer::backpropagateOutputPermutation(qc);
  EXPECT_EQ(qc.initialLayout[0], qc.outputPermutation[0]);
  EXPECT_EQ(qc.initialLayout[1], 0);
}

TEST(BackpropagateOutputPermutation, FullySpecifiedWithSWAP) {
  // i *  *      i 1  0
  //   x--x  ->    x--x
  // o 0  1      o 0  1

  auto qc = QuantumComputation(2);
  qc.outputPermutation.clear();
  qc.outputPermutation[0] = 0;
  qc.outputPermutation[1] = 1;
  qc.swap(0, 1);
  CircuitOptimizer::backpropagateOutputPermutation(qc);
  EXPECT_EQ(qc.initialLayout[0], qc.outputPermutation[1]);
  EXPECT_EQ(qc.initialLayout[1], qc.outputPermutation[0]);
}

TEST(BackpropagateOutputPermutation, PartiallySpecifiedWithSWAP) {
  // i *  *      i 1  0
  //   x--x  ->    x--x
  // o 0  *      o 0  *

  auto qc = QuantumComputation(2);
  qc.outputPermutation.clear();
  qc.outputPermutation[0] = 0;
  qc.swap(0, 1);
  CircuitOptimizer::backpropagateOutputPermutation(qc);
  EXPECT_EQ(qc.initialLayout[0], 1);
  EXPECT_EQ(qc.initialLayout[1], qc.outputPermutation[0]);
}

TEST(BackpropagateOutputPermutation, PartiallySpecifiedWithSWAP2) {
  // i *  *      i 1  0
  //   x--x  ->    x--x
  // o *  1      o *  1

  auto qc = QuantumComputation(2);
  qc.outputPermutation.clear();
  qc.outputPermutation[1] = 1;
  qc.swap(0, 1);
  CircuitOptimizer::backpropagateOutputPermutation(qc);
  EXPECT_EQ(qc.initialLayout[0], qc.outputPermutation[1]);
  EXPECT_EQ(qc.initialLayout[1], 0);
}

TEST(BackpropagateOutputPermutation, PartiallySpecifiedWithSWAP3) {
  // i *  *      i 0  1
  //   x--x  ->    x--x
  // o *  *      o *  *

  auto qc = QuantumComputation(2);
  qc.outputPermutation.clear();
  qc.swap(0, 1);
  CircuitOptimizer::backpropagateOutputPermutation(qc);
  EXPECT_EQ(qc.initialLayout[0], 0);
  EXPECT_EQ(qc.initialLayout[1], 1);
}

TEST(BackpropagateOutputPermutation, CompoundOperation) {
  // i *  *      i 0  1
  //   ----        ----
  //   x--x  ->    x--x
  //   ----        ----
  // o 1  0      o 1  0
  auto qc = QuantumComputation(2);
  qc.outputPermutation.clear();
  qc.outputPermutation[0] = 1;
  qc.outputPermutation[1] = 0;

  auto qc2 = QuantumComputation(2);
  qc2.swap(0, 1);
  qc.emplace_back(qc2.asCompoundOperation());
  CircuitOptimizer::backpropagateOutputPermutation(qc);
  EXPECT_EQ(qc.initialLayout[0], qc.outputPermutation[1]);
  EXPECT_EQ(qc.initialLayout[1], qc.outputPermutation[0]);
}

TEST(BackpropagateOutputPermutation, PartiallySpecifiedNotAMissingQubit) {
  // i *  *  *      i 2  0  1
  //   x--x  |  ->    x--x  |
  // o 0  *  1      o 0  *  1

  auto qc = QuantumComputation(3);
  qc.outputPermutation.clear();
  qc.outputPermutation[0] = 0;
  qc.outputPermutation[2] = 1;
  qc.swap(0, 1);
  CircuitOptimizer::backpropagateOutputPermutation(qc);
  EXPECT_EQ(qc.initialLayout[0], 2);
  EXPECT_EQ(qc.initialLayout[1], 0);
  EXPECT_EQ(qc.initialLayout[2], 1);
}

TEST(BackpropagateOutputPermutation, PartiallySpecifiedNotAMissingQubit2) {
  // i *  *  *      i 0  2  1
  //   x--x  |  ->    x--x  |
  // o *  0  1      o *  0  1

  auto qc = QuantumComputation(3);
  qc.outputPermutation.clear();
  qc.outputPermutation[1] = 0;
  qc.outputPermutation[2] = 1;
  qc.swap(0, 1);
  CircuitOptimizer::backpropagateOutputPermutation(qc);
  EXPECT_EQ(qc.initialLayout[0], 0);
  EXPECT_EQ(qc.initialLayout[1], 2);
  EXPECT_EQ(qc.initialLayout[2], 1);
}

TEST(BackpropagateOutputPermutation, PartiallySpecifiedNotAMissingQubit3) {
  // i *  *  *  *      i 0  3  2  1
  //   x-----x  |  ->    x-----x  |
  // o *  *  0  1      o *  *  0  1

  auto qc = QuantumComputation(4);
  qc.outputPermutation.clear();
  qc.outputPermutation[2] = 0;
  qc.outputPermutation[3] = 1;
  qc.swap(0, 2);
  CircuitOptimizer::backpropagateOutputPermutation(qc);
  EXPECT_EQ(qc.initialLayout[0], 0);
  EXPECT_EQ(qc.initialLayout[1], 3);
  EXPECT_EQ(qc.initialLayout[2], 2);
  EXPECT_EQ(qc.initialLayout[3], 1);
}

} // namespace qc
