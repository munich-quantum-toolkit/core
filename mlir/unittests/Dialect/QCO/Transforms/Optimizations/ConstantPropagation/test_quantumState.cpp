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
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/QuantumState.hpp"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace mlir::qco;

class QuantumStateTest : public testing::Test {
protected:
  mlir::MLIRContext context;
  QCOProgramBuilder programBuilder;
  QCOProgramBuilder referenceBuilder;
  std::vector<unsigned int> fourQubits = {0, 1, 2, 3};
  std::vector<unsigned int> vectorZero = {0};
  std::vector<unsigned int> vectorOne = {1};
  std::vector<unsigned int> vectorTwo = {2};
  std::vector<unsigned int> vectorThree = {3};
  std::vector<unsigned int> vectorFour = {4};
  std::vector<unsigned int> vectorZeroOne = {0, 1};
  std::vector<unsigned int> vectorOneThree = {1, 3};
  std::vector<unsigned int> vectorTwoOne = {2, 1};
  std::vector<unsigned int> vectorZeroTwoFour = {0, 2, 4};

  IdOp idOp;
  HOp hOp;
  XOp xOp;
  ZOp zOp;
  SOp sOp;
  SXOp sxOp;
  SXdgOp sxdgOp;
  TdgOp tdgOp;
  UOp uOp;
  DCXOp dcxOp;
  SWAPOp swapOp;
  iSWAPOp iSwapOp;
  RZOp rzOp;
  POp pOp;
  ECROp ecrOp;
  RXXOp rxxOp;
  RYYOp ryyOp;
  RZXOp rzxOp;
  RZZOp rzzOp;
  XXPlusYYOp xxPlusyyOp;
  XXMinusYYOp xxMinusyyOp;

  QuantumStateTest() : programBuilder(&context), referenceBuilder(&context) {}

  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<QCODialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    programBuilder.initialize();
    referenceBuilder.initialize();

    auto q = programBuilder.allocQubitRegister(4);
    idOp = IdOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                        q[0]);
    hOp = HOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    xOp = XOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    zOp = ZOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    sOp = SOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    sxOp = SXOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                        q[0]);
    sxdgOp = SXdgOp::create(programBuilder, programBuilder.getLoc(),
                            q[0].getType(), q[0]);
    uOp = UOp::create(programBuilder, programBuilder.getLoc(), {q[0].getType()},
                      {q[0], q[1], q[2], q[3]});
    tdgOp = TdgOp::create(programBuilder, programBuilder.getLoc(),
                          q[0].getType(), q[0]);
    dcxOp = DCXOp::create(programBuilder, programBuilder.getLoc(),
                          q[0].getType(), q[1].getType(), q[0], q[1]);
    swapOp = SWAPOp::create(programBuilder, programBuilder.getLoc(),
                            q[0].getType(), q[1].getType(), q[0], q[1]);
    iSwapOp = iSWAPOp::create(programBuilder, programBuilder.getLoc(),
                              q[0].getType(), q[1].getType(), q[0], q[1]);
    rzOp = RZOp::create(programBuilder, programBuilder.getLoc(),
                        {q[0].getType()}, {q[0], q[1]});
    pOp = POp::create(programBuilder, programBuilder.getLoc(), {q[0].getType()},
                      {q[0], q[1]});
    ecrOp = ECROp::create(programBuilder, programBuilder.getLoc(),
                          q[0].getType(), q[1].getType(), q[0], q[1]);
    rxxOp = RXXOp::create(programBuilder, programBuilder.getLoc(),
                          q[0].getType(), q[1].getType(), q[0], q[1], q[2]);
    ryyOp = RYYOp::create(programBuilder, programBuilder.getLoc(),
                          q[0].getType(), q[1].getType(), q[0], q[1], q[2]);
    rzxOp = RZXOp::create(programBuilder, programBuilder.getLoc(),
                          q[0].getType(), q[1].getType(), q[0], q[1], q[2]);
    rzzOp = RZZOp::create(programBuilder, programBuilder.getLoc(),
                          q[0].getType(), q[1].getType(), q[0], q[1], q[2]);
    xxPlusyyOp = XXPlusYYOp::create(programBuilder, programBuilder.getLoc(),
                                    q[0].getType(), q[1].getType(), q[0], q[1],
                                    q[2], q[3]);
    xxMinusyyOp = XXMinusYYOp::create(programBuilder, programBuilder.getLoc(),
                                      q[0].getType(), q[1].getType(), q[0],
                                      q[1], q[2], q[3]);
    ;
  }

  void TearDown() override {}
};

TEST_F(QuantumStateTest, applyHGate) {
  auto qState = QuantumState(vectorZero, 4);
  qState.propagateGate(hOp.getOperation(), vectorZero);

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0> -> 0.71, |1> -> 0.71"));
}

TEST_F(QuantumStateTest, applyHGateToThirdQubit) {
  auto qState = QuantumState(fourQubits, 4);
  qState.propagateGate(hOp.getOperation(), vectorTwo);

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0000> -> 0.71, |0100> -> 0.71"));
}

TEST_F(QuantumStateTest, applyHHGateToThirdQubit) {
  auto qState = QuantumState(fourQubits, 4);
  qState.propagateGate(hOp.getOperation(), vectorTwo);
  qState.propagateGate(hOp.getOperation(), vectorTwo);

  EXPECT_THAT(qState.toString(), testing::HasSubstr("|0000> -> 1"));
}

TEST_F(QuantumStateTest, applyHZGateToThirdQubit) {
  auto qState = QuantumState(fourQubits, 4);
  qState.propagateGate(hOp.getOperation(), vectorTwo);
  qState.propagateGate(zOp.getOperation(), vectorTwo);

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0000> -> 0.71, |0100> -> -0.71"));
}

TEST_F(QuantumStateTest, applyHZHGateToThirdQubit) {
  auto qState = QuantumState(fourQubits, 4);
  qState.propagateGate(hOp.getOperation(), vectorTwo);
  qState.propagateGate(zOp.getOperation(), vectorTwo);
  qState.propagateGate(hOp.getOperation(), vectorTwo);

  EXPECT_THAT(qState.toString(), testing::HasSubstr("|0100> -> 1"));
}

TEST_F(QuantumStateTest, applyHGatesToTwoQubits) {
  auto qState = QuantumState(fourQubits, 4);
  qState.propagateGate(hOp.getOperation(), vectorTwo);
  qState.propagateGate(xOp.getOperation(), vectorZero);

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0001> -> 0.71, |0101> -> 0.71"));
}

TEST_F(QuantumStateTest, applyParametrizedGateToThirdQubit) {
  std::vector<double> params = {1, 0.5, 2};
  auto qState = QuantumState(fourQubits, 4);
  qState.propagateGate(hOp.getOperation(), vectorTwo);
  qState.propagateGate(uOp.getOperation(), vectorTwo, {}, params);

  EXPECT_THAT(
      qState.toString(),
      testing::HasSubstr("|0000> -> 0.76 - i0.31, |0100> -> -0.20 + i0.53"));
}

TEST_F(QuantumStateTest, applyTwoQubitGate) {
  auto qState = QuantumState(fourQubits, 4);
  qState.propagateGate(hOp.getOperation(), vectorOne);
  qState.propagateGate(sOp.getOperation(), vectorOne);
  qState.propagateGate(hOp.getOperation(), vectorTwo);
  qState.propagateGate(tdgOp.getOperation(), vectorTwo);
  qState.propagateGate(dcxOp.getOperation(), vectorTwoOne);

  EXPECT_THAT(
      qState.toString(),
      testing::HasSubstr("|0000> -> 0.50, |0010> -> 0.35 - i0.35, "
                         "|0100> -> 0.35 + i0.35, |0110> -> 0.00 + i0.50"));
}

TEST_F(QuantumStateTest, applyTwoQubitGateReversedOrd) {
  std::vector<unsigned int> vectorOneTwo = {1, 2};
  auto qState = QuantumState(fourQubits, 4);
  qState.propagateGate(hOp.getOperation(), vectorOne);
  qState.propagateGate(sOp.getOperation(), vectorOne);
  qState.propagateGate(hOp.getOperation(), vectorTwo);
  qState.propagateGate(tdgOp.getOperation(), vectorTwo);
  qState.propagateGate(dcxOp.getOperation(), vectorOneTwo);

  EXPECT_THAT(
      qState.toString(),
      testing::HasSubstr("|0000> -> 0.50, |0010> -> 0.35 + i0.35, |0100> -> "
                         "0.00 + i0.50, |0110> -> 0.35 - i0.35"));
}

TEST_F(QuantumStateTest, applySwapGate) {
  std::vector<unsigned int> vectorOneThree = {1, 3};
  auto qState = QuantumState(fourQubits, 4);
  qState.propagateGate(hOp.getOperation(), vectorOne);
  qState.propagateGate(swapOp.getOperation(), vectorOneThree);

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0000> -> 0.71, |1000> -> 0.71"));
}

TEST_F(QuantumStateTest, applyControlledGate1) {
  auto qState = QuantumState(fourQubits, 4);
  qState.propagateGate(hOp.getOperation(), vectorOne);
  qState.propagateGate(xOp.getOperation(), vectorThree);
  qState.propagateGate(xOp.getOperation(), vectorThree, vectorOne);

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0010> -> 0.71, |1000> -> 0.71"));
}

TEST_F(QuantumStateTest, applyControlledGate2) {
  auto qState = QuantumState(fourQubits, 8);
  qState.propagateGate(hOp.getOperation(), vectorZero);
  qState.propagateGate(hOp.getOperation(), vectorOne);
  qState.propagateGate(hOp.getOperation(), vectorTwo);
  qState.propagateGate(xOp.getOperation(), vectorThree, vectorZeroOne);

  EXPECT_THAT(
      qState.toString(),
      testing::HasSubstr(
          "|0000> -> 0.35, |0001> -> 0.35, |0010> -> 0.35, |0100> -> 0.35, "
          "|0101> -> 0.35, |0110> -> 0.35, |1011> -> 0.35, |1111> -> 0.35"));
}

TEST_F(QuantumStateTest, applyControlledTwoQubitGate) {
  auto qState = QuantumState(fourQubits, 4);
  qState.propagateGate(hOp.getOperation(), vectorThree);
  qState.propagateGate(hOp.getOperation(), vectorTwo);
  qState.propagateGate(swapOp.getOperation(), vectorTwoOne, vectorThree);

  EXPECT_THAT(
      qState.toString(),
      testing::HasSubstr(
          "|0000> -> 0.50, |0100> -> 0.50, |1000> -> 0.50, |1010> -> 0.50"));
}

TEST_F(QuantumStateTest, propagateGateCheckErrorIfTwoManyAmplitudesAreNonzero) {
  auto qState = QuantumState(fourQubits, 2);
  qState.propagateGate(hOp.getOperation(), vectorThree);
  qState.propagateGate(xOp.getOperation(), vectorTwo, vectorThree);

  EXPECT_THROW(qState.propagateGate(hOp.getOperation(), vectorTwo);
               , std::domain_error);
}

TEST_F(QuantumStateTest, doMeasurementWithZeroResult) {
  auto qState = QuantumState(vectorZero, 2);
  const auto [states, availableStates] = qState.measureQubit(0);

  EXPECT_TRUE(availableStates.size() == 1);
  auto [probability, qs] = states.at(0);
  EXPECT_TRUE(qState == *qs.get());
  EXPECT_DOUBLE_EQ(probability, 1);
}

TEST_F(QuantumStateTest, doMeasurementWithOneResult) {
  auto qState = QuantumState(vectorZero, 2);
  qState.propagateGate(xOp.getOperation(), vectorZero);
  const auto [states, availableStates] = qState.measureQubit(0);

  EXPECT_TRUE(availableStates.size() == 1);
  auto [probability, qs] = states.at(1);
  EXPECT_TRUE(qState == *qs.get());
  EXPECT_DOUBLE_EQ(probability, 1);
}

TEST_F(QuantumStateTest, doMeasurementWithTwoResults) {
  auto qState = QuantumState(vectorZeroOne, 2);
  qState.propagateGate(hOp.getOperation(), vectorZero);
  qState.propagateGate(xOp.getOperation(), vectorOne, vectorZero);
  const auto [states, availableStates] = qState.measureQubit(0);

  auto const zeroReference = QuantumState(vectorZeroOne, 2);
  auto oneReference = QuantumState(vectorZeroOne, 2);
  oneReference.propagateGate(xOp.getOperation(), vectorZero);
  oneReference.propagateGate(xOp.getOperation(), vectorOne);

  EXPECT_TRUE(availableStates.size() == 2);
  auto [probabilityZero, qsZero] = states.at(0);
  EXPECT_TRUE(zeroReference == *qsZero.get());
  EXPECT_DOUBLE_EQ(probabilityZero, 0.5);
  auto [probabilityOne, qsOne] = states.at(1);
  EXPECT_TRUE(oneReference == *qsOne.get());
  EXPECT_DOUBLE_EQ(probabilityOne, 0.5);
}

TEST_F(QuantumStateTest, doResetWithOnlyZeros) {
  auto qState = QuantumState(vectorZero, 2);
  const auto [states, availableStates] = qState.resetQubit(0);

  EXPECT_TRUE(availableStates.size() == 1);
  auto [probability, qs] = states.at(0);
  EXPECT_TRUE(qState == *qs.get());
  EXPECT_DOUBLE_EQ(probability, 1);
}

TEST_F(QuantumStateTest, doResetWithOnlyOnes) {
  auto qState = QuantumState(vectorZero, 2);
  qState.propagateGate(xOp.getOperation(), vectorZero);
  const auto [states, availableStates] = qState.resetQubit(0);

  auto const refState = QuantumState(vectorZero, 2);

  EXPECT_TRUE(availableStates.size() == 1);
  auto [probability, qs] = states.at(1);
  EXPECT_TRUE(refState == *qs.get());
  EXPECT_DOUBLE_EQ(probability, 1);
}

TEST_F(QuantumStateTest, doResetWithZerosAndOnes) {
  auto qState = QuantumState(vectorZeroOne, 2);
  qState.propagateGate(hOp.getOperation(), vectorZero);
  qState.propagateGate(xOp.getOperation(), vectorOne, vectorZero);
  const auto [states, availableStates] = qState.resetQubit(0);

  auto const zeroReference = QuantumState(vectorZeroOne, 2);
  auto oneReference = QuantumState(vectorZeroOne, 2);
  oneReference.propagateGate(xOp.getOperation(), vectorOne);

  EXPECT_TRUE(availableStates.size() == 2);
  auto [probabilityZero, qsZero] = states.at(0);
  EXPECT_DOUBLE_EQ(probabilityZero, 0.5);
  auto [probabilityOne, qsOne] = states.at(1);
  EXPECT_DOUBLE_EQ(probabilityOne, 0.5);
  EXPECT_TRUE(*qsZero.get() == zeroReference);
  EXPECT_TRUE(*qsOne.get() == oneReference);
}

TEST_F(QuantumStateTest, unifyTwoQuantumStates) {
  auto qState1 = QuantumState(vectorZeroTwoFour, 10);
  qState1.propagateGate(hOp.getOperation(), vectorFour);
  qState1.propagateGate(xOp.getOperation(), vectorTwo, vectorFour);
  qState1.propagateGate(xOp.getOperation(), vectorZero, vectorTwo);

  auto qState2 = QuantumState(vectorOneThree, 10);
  qState2.propagateGate(hOp.getOperation(), vectorThree);
  qState2.propagateGate(xOp.getOperation(), vectorOne, vectorThree);

  const QuantumState unified = qState1.unify(qState2);

  EXPECT_THAT(unified.toString(),
              testing::HasSubstr("|00000> -> 0.50, |01010> -> 0.50, "
                                 "|10101> -> 0.50, |11111> -> 0.50"));
}

TEST_F(QuantumStateTest, unifyTooLargeQuantumStates) {
  auto qState1 = QuantumState(vectorZeroTwoFour, 3);
  qState1.propagateGate(hOp.getOperation(), vectorFour);
  qState1.propagateGate(xOp.getOperation(), vectorTwo, vectorFour);
  qState1.propagateGate(xOp.getOperation(), vectorZero, vectorTwo);

  auto qState2 = QuantumState(vectorOneThree, 3);
  qState2.propagateGate(hOp.getOperation(), vectorThree);
  qState2.propagateGate(xOp.getOperation(), vectorOne, vectorThree);

  EXPECT_THROW(auto qs = qState1.unify(qState2), std::domain_error);
}

TEST_F(QuantumStateTest, applyVariousGates) {
  std::vector paramsZero = {1.0};
  std::vector paramsOne = {2.7};
  std::vector paramsTwo = {1.3, 2.0};
  std::vector paramsThree = {-1.3, 2.0};
  auto qState = QuantumState(vectorZeroOne, 4);
  qState.propagateGate(idOp.getOperation(), vectorZero);
  qState.propagateGate(sxOp.getOperation(), vectorZero);
  qState.propagateGate(sxdgOp.getOperation(), vectorOne);
  qState.propagateGate(iSwapOp.getOperation(), vectorZeroOne);
  qState.propagateGate(rzOp.getOperation(), vectorOne, {}, paramsZero);
  qState.propagateGate(pOp.getOperation(), vectorZero, {}, paramsOne);
  qState.propagateGate(ecrOp.getOperation(), vectorZeroOne);
  qState.propagateGate(rxxOp.getOperation(), vectorZeroOne, {}, paramsZero);
  qState.propagateGate(ryyOp.getOperation(), vectorZeroOne, {}, paramsZero);
  qState.propagateGate(rzxOp.getOperation(), vectorZeroOne, {}, paramsZero);
  qState.propagateGate(rzzOp.getOperation(), vectorZeroOne, {}, paramsZero);
  qState.propagateGate(xxMinusyyOp.getOperation(), vectorZeroOne, {},
                       paramsTwo);
  qState.propagateGate(xxPlusyyOp.getOperation(), vectorZeroOne, {},
                       paramsThree);

  EXPECT_THAT(
      qState.toString(),
      testing::HasSubstr("|00> -> -0.27 - i0.25, |01> -> 0.12 - i0.39, |10> "
                         "-> -0.62 + i0.43, |11> -> -0.15 - i0.32"));
}

} // namespace
