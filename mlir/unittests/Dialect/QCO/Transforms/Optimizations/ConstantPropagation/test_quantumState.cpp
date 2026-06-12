/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/QuantumState.hpp"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <memory>
#include <mlir/Dialect/Arith/IR/ArithOpsDialect.h.inc>
#include <mlir/Dialect/Func/IR/FuncOpsDialect.h.inc>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace mlir::qco;

class QuantumStateTest : public ::testing::Test {
protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(QuantumStateTest, ApplyHGate) {
  mlir::MLIRContext context;
  QCOProgramBuilder programBuilder(&context);
  QCOProgramBuilder referenceBuilder(&context);

  mlir::DialectRegistry registry;
  registry.insert<QCODialect, mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  programBuilder.initialize();
  referenceBuilder.initialize();

  auto q = programBuilder.allocQubitRegister(1);

  auto h = HOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(), q[0]);

  std::vector<unsigned int> qubits = {0};
  QuantumState qState = QuantumState(qubits, 4);
  qState.propagateGate(h.getOperation(), qubits);

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0> -> 0.71, |1> -> 0.71"));
}

TEST_F(QuantumStateTest, ApplyHGateToThirdQubit) {
  std::vector<unsigned int> qubits = {0, 1, 2, 3};
  std::vector<unsigned int> targets = {2};
  QuantumState qState = QuantumState(qubits, 4);
  qState.propagateGate(HOp(), targets);

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0000> -> 0.71, |0100> -> 0.71"));
}

TEST_F(QuantumStateTest, ApplyHHGateToThirdQubit) {
  std::vector<unsigned int> qubits = {0, 1, 2, 3};
  std::vector<unsigned int> targets = {2};
  QuantumState qState = QuantumState(qubits, 4);
  qState.propagateGate(HOp(), targets);
  qState.propagateGate(HOp(), targets);

  EXPECT_THAT(qState.toString(), testing::HasSubstr("|0000> -> 1"));
}

TEST_F(QuantumStateTest, ApplyHZGateToThirdQubit) {
  std::vector<unsigned int> qubits = {0, 1, 2, 3};
  std::vector<unsigned int> targets = {2};
  QuantumState qState = QuantumState(qubits, 4);
  qState.propagateGate(HOp(), targets);
  qState.propagateGate(ZOp(), targets);

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0000> -> 0.71, |0100> -> -0.71"));
}

TEST_F(QuantumStateTest, ApplyHZHGateToThirdQubit) {
  std::vector<unsigned int> qubits = {0, 1, 2, 3};
  std::vector<unsigned int> targets = {2};
  QuantumState qState = QuantumState(qubits, 4);
  qState.propagateGate(HOp(), targets);
  qState.propagateGate(ZOp(), targets);
  qState.propagateGate(HOp(), targets);

  EXPECT_THAT(qState.toString(), testing::HasSubstr("|0100> -> 1"));
}

TEST_F(QuantumStateTest, ApplyHGatesToTwoQubits) {
  std::vector<unsigned int> qubits = {0, 1, 2, 3};
  std::vector<unsigned int> targetZero = {0};
  std::vector<unsigned int> targetTwo = {2};
  QuantumState qState = QuantumState(qubits, 4);
  qState.propagateGate(HOp(), targetTwo);
  qState.propagateGate(XOp(), targetZero);

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0001> -> 0.71, |0101> -> 0.71"));
}

TEST_F(QuantumStateTest, ApplyParametrizedGateToThirdQubit) {
  std::vector<unsigned int> qubits = {0, 1, 2, 3};
  std::vector<unsigned int> target = {2};
  std::vector<double> params = {1, 0.5, 2};
  QuantumState qState = QuantumState(qubits, 4);
  qState.propagateGate(HOp(), target);
  qState.propagateGate(UOp(), target, {}, params);

  EXPECT_THAT(
      qState.toString(),
      testing::HasSubstr("|0000> -> 0.76 - i0.31, |0100> -> -0.20 + i0.53"));
}

TEST_F(QuantumStateTest, ApplyTwoQubitGate) {
  std::vector<unsigned int> qubits = {0, 1, 2, 3};
  std::vector<unsigned int> targetOne = {1};
  std::vector<unsigned int> targetTwo = {2};
  std::vector<unsigned int> targets = {2, 1};
  QuantumState qState = QuantumState(qubits, 4);
  qState.propagateGate(HOp(), targetOne);
  qState.propagateGate(SOp(), targetOne);
  qState.propagateGate(HOp(), targetTwo);
  qState.propagateGate(TdgOp(), targetTwo);
  qState.propagateGate(DCXOp(), targets);

  EXPECT_THAT(
      qState.toString(),
      testing::HasSubstr("|0000> -> 0.35 + i0.35, |0010> -> 0.35 - i0.35, "
                         "|0100> -> 0.50, |0110> -> 0.00 + i0.50"));
}

TEST_F(QuantumStateTest, ApplyTwoQubitGateReversedOrd) {
  std::vector<unsigned int> qubits = {0, 1, 2, 3};
  std::vector<unsigned int> targetOne = {1};
  std::vector<unsigned int> targetTwo = {2};
  std::vector<unsigned int> targets = {1, 2};
  QuantumState qState = QuantumState(qubits, 4);
  qState.propagateGate(HOp(), targetOne);
  qState.propagateGate(SOp(), targetOne);
  qState.propagateGate(HOp(), targetTwo);
  qState.propagateGate(TdgOp(), targetTwo);
  qState.propagateGate(DCXOp(), targets);

  EXPECT_THAT(
      qState.toString(),
      testing::HasSubstr("|0000> -> 0.35 + i0.35, |0010> -> 0.50, "
                         "|0100> -> 0.00 + i0.50, |0110> -> 0.35 - i0.35"));
}

TEST_F(QuantumStateTest, ApplySwapGate) {
  std::vector<unsigned int> qubits = {0, 1, 2, 3};
  std::vector<unsigned int> targetOne = {1};
  std::vector<unsigned int> targets = {1, 3};
  QuantumState qState = QuantumState(qubits, 4);
  qState.propagateGate(HOp(), targetOne);
  qState.propagateGate(SWAPOp(), targets);

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0000> -> 0.71, |1000> -> 0.71"));
}

TEST_F(QuantumStateTest, ApplyControlledGate) {
  std::vector<unsigned int> qubits = {0, 1, 2, 3};
  std::vector<unsigned int> targetOne = {1};
  std::vector<unsigned int> targetThree = {3};
  QuantumState qState = QuantumState(qubits, 4);
  qState.propagateGate(HOp(), targetOne);
  qState.propagateGate(XOp(), targetThree);
  qState.propagateGate(XOp(), targetThree, targetOne);

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0010> -> 0.71, |1000> -> 0.71"));
}

TEST_F(QuantumStateTest, ApplyPosNegControlledGate) {
  std::vector<unsigned int> qubits = {0, 1, 2, 3};
  std::vector<unsigned int> targetZero = {0};
  std::vector<unsigned int> targetOne = {1};
  std::vector<unsigned int> targetTwo = {2};
  std::vector<unsigned int> targetThree = {3};
  std::vector<unsigned int> targets = {0, 1};
  std::vector<double> params = {2.0};
  QuantumState qState = QuantumState(qubits, 8);
  qState.propagateGate(HOp(), targetZero);
  qState.propagateGate(HOp(), targetOne);
  qState.propagateGate(HOp(), targetTwo);
  qState.propagateGate(XOp(), targetThree, targets, params);

  EXPECT_THAT(
      qState.toString(),
      testing::HasSubstr("|0000> -> 0.35, |0001> -> 0.35, |0010> -> 0.35, "
                         "|0100> -> 0.35, |0101> -> 0.35, |0110> -> 0.35, "
                         "|0111> -> 0.35, |1011> -> 0.35"));
}

TEST_F(QuantumStateTest, ApplyControlledTwoQubitGate) {
  std::vector<unsigned int> qubits = {0, 1, 2, 3};
  std::vector<unsigned int> targetTwo = {2};
  std::vector<unsigned int> targetThree = {3};
  std::vector<unsigned int> targets = {2, 1};
  QuantumState qState = QuantumState(qubits, 4);
  qState.propagateGate(HOp(), targetThree);
  qState.propagateGate(HOp(), targetTwo);
  qState.propagateGate(SWAPOp(), targets, targetThree);

  EXPECT_THAT(
      qState.toString(),
      testing::HasSubstr(
          "|0000> -> 0.50, |0100> -> 0.50, |1000> -> 0.50, |1010> -> 0.50"));
}

TEST_F(QuantumStateTest, propagateGateCheckErrorIfTwoManyAmplitudesAreNonzero) {
  std::vector<unsigned int> qubits = {0, 1, 2, 3};
  std::vector<unsigned int> targetTwo = {2};
  std::vector<unsigned int> targetThree = {3};
  QuantumState qState = QuantumState(qubits, 2);
  qState.propagateGate(HOp(), targetThree);
  qState.propagateGate(XOp(), targetTwo, targetThree);

  EXPECT_THROW(qState.propagateGate(HOp(), targetTwo);, std::domain_error);
}

TEST_F(QuantumStateTest, doMeasurementWithZeroResult) {
  std::vector<unsigned int> qubit = {0};
  QuantumState qState = QuantumState(qubit, 2);
  MeasurementResult const res = qState.measureQubit(0);

  EXPECT_TRUE(res.size == 1);
  auto [probability, qs] = res.states.at(0);
  EXPECT_TRUE(qState == *qs.get());
  EXPECT_DOUBLE_EQ(probability, 1);
}

TEST_F(QuantumStateTest, doMeasurementWithOneResult) {
  std::vector<unsigned int> qubit = {0};
  std::vector<unsigned int> qubitsOne = {0, 2, 4};
  std::vector<unsigned int> qubitsTwo = {1, 3};
  std::vector<unsigned int> targetZero = {0};
  std::vector<unsigned int> targetOne = {1};
  std::vector<unsigned int> targetTwo = {2};
  std::vector<unsigned int> targetThree = {3};
  std::vector<unsigned int> targetFour = {4};
  QuantumState qState = QuantumState(qubit, 2);
  qState.propagateGate(XOp(), qubit);
  MeasurementResult const res = qState.measureQubit(0);

  EXPECT_TRUE(res.size == 1);
  auto [probability, qs] = res.states.at(1);
  EXPECT_TRUE(qState == *qs.get());
  EXPECT_DOUBLE_EQ(probability, 1);
}

TEST_F(QuantumStateTest, doMeasurementWithTwoResults) {
  std::vector<unsigned int> qubits = {0, 1};
  std::vector<unsigned int> targetZero = {0};
  std::vector<unsigned int> targetOne = {1};
  QuantumState qState = QuantumState(qubits, 2);
  qState.propagateGate(HOp(), targetZero);
  qState.propagateGate(XOp(), targetOne, targetZero);
  MeasurementResult const res = qState.measureQubit(0);

  QuantumState const zeroReference = QuantumState(qubits, 2);
  QuantumState oneReference = QuantumState(qubits, 2);
  oneReference.propagateGate(XOp(), targetZero);
  oneReference.propagateGate(XOp(), targetOne);

  EXPECT_TRUE(res.size == 2);
  auto [probabilityZero, qsZero] = res.states.at(0);
  EXPECT_TRUE(zeroReference == *qsZero.get());
  EXPECT_DOUBLE_EQ(probabilityZero, 0.5);
  auto [probabilityOne, qsOne] = res.states.at(1);
  EXPECT_TRUE(oneReference == *qsOne.get());
  EXPECT_DOUBLE_EQ(probabilityOne, 0.5);
}

TEST_F(QuantumStateTest, doResetWithOnlyZeros) {
  std::vector<unsigned int> qubit = {0};
  QuantumState qState = QuantumState(qubit, 2);
  MeasurementResult const res = qState.resetQubit(0);

  EXPECT_TRUE(res.size == 1);
  auto [probability, qs] = res.states.at(0);
  EXPECT_TRUE(qState == *qs.get());
  EXPECT_DOUBLE_EQ(probability, 1);
}

TEST_F(QuantumStateTest, doResetWithOnlyOnes) {
  std::vector<unsigned int> qubit = {0};
  QuantumState qState = QuantumState(qubit, 2);
  qState.propagateGate(XOp(), qubit);
  MeasurementResult const res = qState.resetQubit(0);

  QuantumState const refState = QuantumState(qubit, 2);

  EXPECT_TRUE(res.size == 1);
  auto [probability, qs] = res.states.at(1);
  EXPECT_TRUE(refState == *qs.get());
  EXPECT_DOUBLE_EQ(probability, 1);
}

TEST_F(QuantumStateTest, doResetWithZerosAndOnes) {
  std::vector<unsigned int> qubits = {0, 1};
  std::vector<unsigned int> targetZero = {0};
  std::vector<unsigned int> targetOne = {1};
  QuantumState qState = QuantumState(qubits, 2);
  qState.propagateGate(HOp(), targetZero);
  qState.propagateGate(XOp(), targetOne, targetZero);
  MeasurementResult const res = qState.resetQubit(0);

  QuantumState const zeroReference = QuantumState(targetZero, 2);
  QuantumState oneReference = QuantumState(targetZero, 2);
  oneReference.propagateGate(XOp(), targetOne);

  EXPECT_TRUE(res.size == 2);
  auto [probabilityZero, qsZero] = res.states.at(0);
  EXPECT_DOUBLE_EQ(probabilityZero, 0.5);
  auto [probabilityOne, qsOne] = res.states.at(1);
  EXPECT_DOUBLE_EQ(probabilityOne, 0.5);
  EXPECT_TRUE(*qsZero == oneReference || *qsZero == oneReference);
  EXPECT_TRUE(*qsOne == zeroReference || *qsOne == zeroReference);
}

TEST_F(QuantumStateTest, unifyTwoQuantumStates) {
  std::vector<unsigned int> qubitsOne = {0, 2, 4};
  std::vector<unsigned int> qubitsTwo = {1, 3};
  std::vector<unsigned int> targetZero = {0};
  std::vector<unsigned int> targetOne = {1};
  std::vector<unsigned int> targetTwo = {2};
  std::vector<unsigned int> targetThree = {3};
  std::vector<unsigned int> targetFour = {4};
  QuantumState qState1 = QuantumState(qubitsOne, 10);
  qState1.propagateGate(HOp(), targetFour);
  qState1.propagateGate(XOp(), targetTwo, targetFour);
  qState1.propagateGate(XOp(), targetZero, targetTwo);

  QuantumState qState2 = QuantumState(qubitsTwo, 10);
  qState2.propagateGate(HOp(), targetThree);
  qState2.propagateGate(XOp(), targetOne, targetThree);

  const QuantumState unified = qState1.unify(qState2);

  EXPECT_THAT(unified.toString(),
              testing::HasSubstr("|00000> -> 0.50, |01010> -> 0.50, "
                                 "|10101> -> 0.50, |11111> -> 0.50"));
}

TEST_F(QuantumStateTest, unifyTooLargeQuantumStates) {
  std::vector<unsigned int> qubitsOne = {0, 2, 4};
  std::vector<unsigned int> qubitsTwo = {1, 3};
  std::vector<unsigned int> targetZero = {0};
  std::vector<unsigned int> targetOne = {1};
  std::vector<unsigned int> targetTwo = {2};
  std::vector<unsigned int> targetThree = {3};
  std::vector<unsigned int> targetFour = {4};
  QuantumState qState1 = QuantumState(qubitsOne, 3);
  qState1.propagateGate(HOp(), targetFour);
  qState1.propagateGate(XOp(), targetTwo, targetFour);
  qState1.propagateGate(XOp(), targetZero, targetTwo);

  QuantumState qState2 = QuantumState(qubitsTwo, 3);
  qState2.propagateGate(HOp(), targetThree);
  qState2.propagateGate(XOp(), targetOne, targetThree);

  EXPECT_THROW(auto qs = qState1.unify(qState2), std::domain_error);
}
