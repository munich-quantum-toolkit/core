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
#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/HybridState.hpp"
#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/QuantumState.hpp"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Value.h>

#include <memory>
#include <mlir/Dialect/Func/IR/FuncOpsDialect.h.inc>
#include <vector>

using namespace mlir::qco;

class HybridStateTest : public testing::Test {
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

  HOp hOp;
  XOp xOp;
  ZOp zOp;
  SOp sOp;
  UOp uOp;

  mlir::Value v1;
  mlir::Value v2;
  mlir::Value v3;
  mlir::Value v4;

  HybridStateTest() : programBuilder(&context), referenceBuilder(&context) {}

  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<QCODialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    programBuilder.initialize();
    referenceBuilder.initialize();

    auto q = programBuilder.allocQubitRegister(4);
    hOp = HOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    xOp = XOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    zOp = ZOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    sOp = SOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    uOp = UOp::create(programBuilder, programBuilder.getLoc(), {q[0].getType()},
                      {q[0], q[1], q[2], q[3]});
    v1 = q[0];
    v2 = q[1];
    v3 = q[2];
    v4 = q[3];
  }

  void TearDown() override {}
};

TEST_F(HybridStateTest, ApplyHGate) {
  auto hState = HybridState(vectorZero, 4);
  hState.propagateGate(hOp.getOperation(), vectorZero);

  EXPECT_THAT(hState.toString(),
              testing::HasSubstr("{|0> -> 0.71, |1> -> 0.71}: p = 1.00;"));
}

TEST_F(HybridStateTest, ApplyHGateToThirdQubit) {
  auto hState = HybridState(fourQubits, 4, 0.5);
  hState.propagateGate(hOp.getOperation(), vectorTwo);

  EXPECT_THAT(
      hState.toString(),
      testing::HasSubstr("{|0000> -> 0.71, |0100> -> 0.71}: p = 0.50;"));
}

TEST_F(HybridStateTest, ApplyParametrizedGateToThirdQubit) {
  std::vector params = {v1, v2, v3};
  auto hState = HybridState(fourQubits, 4);
  hState.addIntegerValue(v1, 1);
  hState.addDoubleValue(v2, 0.5);
  hState.addDoubleValue(v3, 2.0);
  hState.propagateGate(hOp.getOperation(), vectorTwo);
  hState.propagateGate(uOp.getOperation(), vectorTwo, {}, {}, {}, params);

  const auto resStr = hState.toString();
  EXPECT_THAT(resStr, testing::HasSubstr(
                          "{|0000> -> 0.76 - i0.31, |0100> -> -0.20 + i0.53}"));
  EXPECT_THAT(resStr, testing::ContainsRegex("integerValue0 = 1"));
  EXPECT_THAT(resStr,
              testing::AnyOf(testing::HasSubstr("doubleValue0 = 2.00"),
                             testing::HasSubstr("doubleValue1 = 2.00")));
  EXPECT_THAT(resStr,
              testing::AnyOf(testing::HasSubstr("doubleValue0 = 0.50"),
                             testing::HasSubstr("doubleValue1 = 0.50")));
}

TEST_F(HybridStateTest, ApplyQuantumControlledGate) {
  auto hState = HybridState(fourQubits, 4);
  hState.propagateGate(hOp.getOperation(), vectorOne);
  hState.propagateGate(xOp.getOperation(), vectorThree);
  hState.propagateGate(xOp.getOperation(), vectorThree, vectorOne);

  EXPECT_THAT(hState.toString(),
              testing::HasSubstr("{|0010> -> 0.71, |1000> -> 0.71}"));
}

TEST_F(HybridStateTest, ApplyClassicalControlledGateThatsFalse) {
  auto hState = HybridState(fourQubits, 4);
  std::vector ctrl = {v1};
  hState.addIntegerValue(v1, 0);
  hState.propagateGate(xOp.getOperation(), vectorThree);
  hState.propagateGate(hOp.getOperation(), vectorOne);
  hState.propagateGate(xOp.getOperation(), vectorThree, vectorOne, ctrl);

  EXPECT_THAT(
      hState.toString(),
      testing::HasSubstr(
          "{|1000> -> 0.71, |1010> -> 0.71}: integerValue0 = 0; p = 1.00"));
}

TEST_F(HybridStateTest, ApplyClassicalControlledGateThatsTrue) {
  auto hState = HybridState(fourQubits, 4);
  std::vector ctrl = {v1};
  hState.addIntegerValue(v1, 1);
  hState.propagateGate(xOp.getOperation(), vectorThree);
  hState.propagateGate(hOp.getOperation(), vectorOne);
  hState.propagateGate(xOp.getOperation(), vectorThree, vectorOne, ctrl);

  EXPECT_THAT(
      hState.toString(),
      testing::HasSubstr(
          "{|0010> -> 0.71, |1000> -> 0.71}: integerValue0 = 1; p = 1.00"));
}

TEST_F(HybridStateTest, ApplyNegClassicalControlledGateThatsFalse) {
  auto hState = HybridState(fourQubits, 4);
  constexpr auto v1 = mlir::Value();
  std::vector ctrl = {v1};
  hState.addIntegerValue(v1, 0);
  hState.propagateGate(xOp.getOperation(), vectorThree);
  hState.propagateGate(hOp.getOperation(), vectorOne);
  hState.propagateGate(xOp.getOperation(), vectorThree, vectorOne, {}, ctrl);

  EXPECT_THAT(
      hState.toString(),
      testing::HasSubstr(
          "{|0010> -> 0.71, |1000> -> 0.71}: integerValue0 = 0; p = 1.00"));
}

TEST_F(HybridStateTest, ApplyTwoTimesClassicalControlledGate) {
  auto hState = HybridState(fourQubits, 4);
  std::vector ctrls = {v1, v2};
  hState.addIntegerValue(v1, 1);
  hState.addIntegerValue(v2, 3);
  hState.propagateGate(xOp.getOperation(), vectorThree);
  hState.propagateGate(hOp.getOperation(), vectorOne);
  hState.propagateGate(xOp.getOperation(), vectorThree, vectorOne, ctrls);

  const auto resStr = hState.toString();
  EXPECT_THAT(resStr, testing::HasSubstr("{|0010> -> 0.71, |1000> -> 0.71}: "));
  EXPECT_THAT(resStr, testing::HasSubstr("; p = 1.00"));
  EXPECT_THAT(resStr, testing::AnyOf(testing::HasSubstr("integerValue0 = 3"),
                                     testing::HasSubstr("integerValue1 = 3")));
  EXPECT_THAT(resStr, testing::AnyOf(testing::HasSubstr("integerValue0 = 1"),
                                     testing::HasSubstr("integerValue1 = 1")));
}

TEST_F(HybridStateTest, handleErrorIfTwoManyAmplitudesAreNonzero) {
  auto hState = HybridState(fourQubits, 2);
  hState.propagateGate(hOp.getOperation(), vectorThree);
  hState.propagateGate(xOp.getOperation(), vectorTwo, vectorThree);
  // Error occures here
  hState.propagateGate(hOp.getOperation(), vectorTwo);
  // Should leave state in TOP
  hState.propagateGate(sOp.getOperation(), vectorZero);

  EXPECT_TRUE(hState.isHybridStateTop());
}

TEST_F(HybridStateTest, doMeasurementWithOneResult) {
  auto hState = HybridState(fourQubits, 4, 0.6);
  hState.addIntegerValue(v1, 0);
  hState.propagateGate(xOp.getOperation(), vectorZero);
  const auto resStates = hState.propagateMeasurement(0, v1);

  EXPECT_TRUE(resStates.size() == 1);
  const auto resHybridState = resStates.at(0);
  EXPECT_THAT(
      resHybridState.toString(),
      testing::HasSubstr("{|0001> -> 1.00}: integerValue0 = 1; p = 0.60"));
}

TEST_F(HybridStateTest, doMeasurementWithTwoResults) {
  auto hState = HybridState(fourQubits, 4, 0.6);
  hState.addIntegerValue(v1, 10);
  hState.propagateGate(hOp.getOperation(), vectorZero);
  hState.propagateGate(xOp.getOperation(), vectorTwo, vectorZero);
  const auto resStates = hState.propagateMeasurement(0, v1);

  EXPECT_TRUE(resStates.size() == 2);
  const auto resStrings =
      resStates.at(0).toString() + resStates.at(1).toString();
  EXPECT_THAT(resStrings, testing::HasSubstr(
                              "{|0000> -> 1.00}: integerValue0 = 0; p = 0.30"));
  EXPECT_THAT(resStrings, testing::HasSubstr(
                              "{|0101> -> 1.00}: integerValue0 = 1; p = 0.30"));
}

TEST_F(HybridStateTest, doMeasurementWithNegClassicalCtrl) {
  auto hState = HybridState(fourQubits, 4, 0.6);
  constexpr auto v1 = mlir::Value();
  hState.addIntegerValue(v1, 0);
  std::vector ctrl = {v1};
  hState.propagateGate(hOp.getOperation(), vectorZero);
  const auto resStates = hState.propagateMeasurement(0, v1, ctrl);

  EXPECT_TRUE(resStates.size() == 1);
  const auto resHybridState = resStates.at(0);
  EXPECT_THAT(
      resHybridState.toString(),
      testing::HasSubstr(
          "{|0000> -> 0.71, |0001> -> 0.71}: integerValue0 = 0; p = 0.60"));
}

TEST_F(HybridStateTest, doMeasurementWithPosNegClassicalCtrl) {
  auto hState = HybridState(fourQubits, 4, 0.6);
  constexpr auto v1 = mlir::Value();
  hState.addIntegerValue(v1, 3);
  std::vector ctrl = {v1};
  hState.propagateGate(hOp.getOperation(), vectorZero);
  const auto resStates = hState.propagateMeasurement(0, v1, {}, ctrl);

  EXPECT_TRUE(resStates.size() == 1);
  const auto resHybridState = resStates.at(0);
  EXPECT_THAT(
      resHybridState.toString(),
      testing::HasSubstr(
          "{|0000> -> 0.71, |0001> -> 0.71}: integerValue0 = 3; p = 0.60"));
}

TEST_F(HybridStateTest, doResetWithOneResult) {
  auto hState = HybridState(vectorZero, 2);
  hState.addIntegerValue(v1, 3);
  std::vector ctrl = {v1};
  hState.propagateGate(xOp.getOperation(), vectorZero);

  const auto resStates = hState.propagateReset(0, ctrl);

  EXPECT_TRUE(resStates.size() == 1);
  const auto resHybridState = resStates.at(0);
  EXPECT_THAT(resHybridState.toString(),
              testing::HasSubstr("{|0> -> 1.00}: integerValue0 = 3; p = 1.00"));
}

TEST_F(HybridStateTest, doResetWithTwoResults) {
  auto hState = HybridState(fourQubits, 2, 0.6);
  hState.addIntegerValue(v1, 0);
  std::vector ctrl = {v1};
  hState.propagateGate(hOp.getOperation(), vectorZero);
  hState.propagateGate(xOp.getOperation(), vectorThree, vectorZero);

  const auto resStates = hState.propagateReset(0, {}, ctrl);

  EXPECT_TRUE(resStates.size() == 2);
  const auto resString =
      resStates.at(0).toString() + resStates.at(1).toString();
  EXPECT_THAT(resString, testing::HasSubstr(
                             "{|0000> -> 1.00}: integerValue0 = 0; p = 0.30"));
  EXPECT_THAT(resString, testing::HasSubstr(
                             "{|1000> -> 1.00}: integerValue0 = 0; p = 0.30"));
}

TEST_F(HybridStateTest, doResetWithNegClassicalCtrl) {
  auto hState = HybridState(fourQubits, 4, 0.6);
  constexpr auto v1 = mlir::Value();
  hState.addIntegerValue(v1, 0);
  std::vector ctrl = {v1};
  hState.propagateGate(hOp.getOperation(), vectorZero);
  const auto resStates = hState.propagateReset(0, ctrl);

  EXPECT_TRUE(resStates.size() == 1);
  const auto resHybridState = resStates.at(0);
  EXPECT_THAT(
      resHybridState.toString(),
      testing::HasSubstr(
          "{|0000> -> 0.71, |0001> -> 0.71}: integerValue0 = 0; p = 0.60"));
}

TEST_F(HybridStateTest, doResetWithPosNegClassicalCtrl) {
  auto hState = HybridState(fourQubits, 4, 0.6);
  constexpr auto v1 = mlir::Value();
  hState.addIntegerValue(v1, 3);
  std::vector ctrl = {v1};
  hState.propagateGate(hOp.getOperation(), vectorZero);
  const auto resStates = hState.propagateReset(0, {}, ctrl);

  EXPECT_TRUE(resStates.size() == 1);
  const auto resHybridState = resStates.at(0);
  EXPECT_THAT(
      resHybridState.toString(),
      testing::HasSubstr(
          "{|0000> -> 0.71, |0001> -> 0.71}: integerValue0 = 3; p = 0.60"));
}

TEST_F(HybridStateTest, doMeasurementOnTop) {
  auto hState = HybridState(fourQubits, 2);
  constexpr auto v1 = mlir::Value();
  hState.propagateGate(hOp.getOperation(), vectorThree);
  hState.propagateGate(xOp.getOperation(), vectorTwo, vectorThree);
  // Error occurs here
  hState.propagateGate(hOp.getOperation(), vectorTwo);
  // Should leave state in TOP
  hState.addIntegerValue(v1, 3);
  hState.propagateMeasurement(0, v1);

  EXPECT_TRUE(hState.isHybridStateTop());
}

TEST_F(HybridStateTest, doResetOnTop) {
  auto hState = HybridState(fourQubits, 2);
  hState.propagateGate(hOp.getOperation(), vectorThree);
  hState.propagateGate(xOp.getOperation(), vectorTwo, vectorThree);
  // Error occures here
  hState.propagateGate(hOp.getOperation(), vectorTwo);
  // Should leave state in TOP
  hState.addIntegerValue(v1, 3);
  hState.propagateReset(0);

  EXPECT_TRUE(hState.isHybridStateTop());
}

TEST_F(HybridStateTest, unifyTwoHybridStates) {

  auto hState1 = HybridState(vectorZeroTwoFour, 10, 0.8);
  hState1.propagateGate(hOp.getOperation(), vectorFour);
  hState1.propagateGate(xOp.getOperation(), vectorTwo, vectorFour);
  hState1.propagateGate(xOp.getOperation(), vectorZero, vectorTwo);
  hState1.addIntegerValue(v1, 4);

  auto hState2 = HybridState(vectorOneThree, 10, 0.5);
  hState2.propagateGate(hOp.getOperation(), vectorThree);
  hState2.propagateGate(xOp.getOperation(), vectorOne, vectorThree);
  hState1.addIntegerValue(v2, 7);
  hState1.addDoubleValue(v3, 4.2);

  const HybridState unified = hState1.unify(hState2);
  const auto resStr = unified.toString();

  EXPECT_THAT(resStr,
              testing::HasSubstr("{|00000> -> 0.50, |01010> -> 0.50, "
                                 "|10101> -> 0.50, |11111> -> 0.50}: "));
  EXPECT_THAT(resStr, testing::HasSubstr("doubleValue0 = 4.20; p = 0.40"));
  EXPECT_THAT(resStr, testing::AnyOf(testing::HasSubstr("integerValue0 = 4"),
                                     testing::HasSubstr("integerValue1 = 4")));
  EXPECT_THAT(resStr, testing::AnyOf(testing::HasSubstr("integerValue0 = 7"),
                                     testing::HasSubstr("integerValue1 = 7")));
}

TEST_F(HybridStateTest, unifyHybridStatesOneWithoutQuantum) {
  auto hState1 = HybridState(fourQubits, 10, 0.8);
  hState1.propagateGate(hOp.getOperation(), vectorOne);
  hState1.addIntegerValue(v1, 4);

  const auto hState2 = HybridState({}, 10, 0.5);
  hState1.addIntegerValue(v2, 7);
  hState1.addDoubleValue(v3, 4.2);

  const HybridState unified = hState1.unify(hState2);
  const auto resStr = unified.toString();

  EXPECT_THAT(resStr, testing::HasSubstr("{|0000> -> 0.71, |0010> -> 0.71}: "));
  EXPECT_THAT(resStr, testing::HasSubstr(", doubleValue0 = 4.20; p = 0.40"));
  EXPECT_THAT(resStr, testing::AnyOf(testing::HasSubstr("integerValue0 = 7"),
                                     testing::HasSubstr("integerValue1 = 7")));
  EXPECT_THAT(resStr, testing::AnyOf(testing::HasSubstr("integerValue0 = 4"),
                                     testing::HasSubstr("integerValue1 = 4")));
}

TEST_F(HybridStateTest, unifyHybridStatesWithoutQuantum) {
  auto hState1 = HybridState({}, 10, 0.8);
  hState1.addIntegerValue(v1, 4);

  const auto hState2 = HybridState({}, 10, 0.5);
  hState1.addIntegerValue(v2, 7);
  hState1.addDoubleValue(v3, 4.2);

  const HybridState unified = hState1.unify(hState2);
  const auto resStr = unified.toString();

  EXPECT_THAT(resStr, testing::HasSubstr("{}: "));
  EXPECT_THAT(resStr, testing::HasSubstr(", doubleValue0 = 4.20; p = 0.40"));
  EXPECT_THAT(resStr, testing::AnyOf(testing::HasSubstr("integerValue0 = 4"),
                                     testing::HasSubstr("integerValue1 = 4")));
  EXPECT_THAT(resStr, testing::AnyOf(testing::HasSubstr("integerValue0 = 7"),
                                     testing::HasSubstr("integerValue1 = 7")));
}

TEST_F(HybridStateTest, unifyTooLargeHybridStates) {
  auto hState1 = HybridState(vectorZeroTwoFour, 3);
  hState1.propagateGate(hOp.getOperation(), vectorFour);
  hState1.propagateGate(xOp.getOperation(), vectorTwo, vectorFour);
  hState1.propagateGate(xOp.getOperation(), vectorZero, vectorTwo);

  auto hState2 = HybridState(vectorOneThree, 3);
  hState2.propagateGate(hOp.getOperation(), vectorThree);
  hState2.propagateGate(xOp.getOperation(), vectorOne, vectorThree);

  const auto hs = hState1.unify(hState2);

  EXPECT_TRUE(hs.isHybridStateTop());
}

TEST_F(HybridStateTest, intOpTwoValueOperation) {
  auto hState = HybridState(fourQubits, 3);
  const auto i1 = programBuilder.getI64IntegerAttr(3);
  const auto i2 = programBuilder.getI64IntegerAttr(9);
  const auto i3 = programBuilder.getI64IntegerAttr(0);

  const mlir::Value val1 = mlir::arith::ConstantOp::create(
      programBuilder, programBuilder.getLoc(), i1);
  const mlir::Value val2 = mlir::arith::ConstantOp::create(
      programBuilder, programBuilder.getLoc(), i2);
  const mlir::Value val3 = mlir::arith::ConstantOp::create(
      programBuilder, programBuilder.getLoc(), i3);

  hState.addIntegerValue(val1, 3);
  hState.addIntegerValue(val2, 9);
  hState.addIntegerValue(val3, 0);

  const auto subIOp = mlir::arith::SubIOp::create(
      programBuilder, programBuilder.getLoc(), val3.getType(), val1, val2);

  hState.propagateClassicalOperation(subIOp, val3, val1, val2);

  const auto resStr = hState.toString();

  EXPECT_THAT(resStr, testing::AnyOf(testing::HasSubstr("integerValue0 = 3"),
                                     testing::HasSubstr("integerValue1 = 3"),
                                     testing::HasSubstr("integerValue2 = 3")));
  EXPECT_THAT(resStr, testing::AnyOf(testing::HasSubstr("integerValue0 = 9"),
                                     testing::HasSubstr("integerValue1 = 9"),
                                     testing::HasSubstr("integerValue2 = 9")));
  EXPECT_THAT(resStr, testing::AnyOf(testing::HasSubstr("integerValue0 = -6"),
                                     testing::HasSubstr("integerValue1 = -6"),
                                     testing::HasSubstr("integerValue2 = -6")));
}

TEST_F(HybridStateTest, intOpThreeValueOperation) {
  auto hState = HybridState(fourQubits, 3);
  const auto i0 = programBuilder.getI64IntegerAttr(0);
  const auto i1 = programBuilder.getI64IntegerAttr(3);
  const auto i2 = programBuilder.getI64IntegerAttr(9);
  const auto i3 = programBuilder.getI64IntegerAttr(0);

  const mlir::Value val0 = mlir::arith::ConstantOp::create(
      programBuilder, programBuilder.getLoc(), i0);
  const mlir::Value val1 = mlir::arith::ConstantOp::create(
      programBuilder, programBuilder.getLoc(), i1);
  const mlir::Value val2 = mlir::arith::ConstantOp::create(
      programBuilder, programBuilder.getLoc(), i2);
  const mlir::Value val3 = mlir::arith::ConstantOp::create(
      programBuilder, programBuilder.getLoc(), i3);

  hState.addIntegerValue(val0, 0);
  hState.addIntegerValue(val1, 3);
  hState.addIntegerValue(val2, 9);
  hState.addIntegerValue(val3, 1);

  const auto selectOp =
      mlir::arith::SelectOp::create(programBuilder, programBuilder.getLoc(),
                                    val3.getType(), val0, val1, val2);
  const auto subIOp = mlir::arith::SubIOp::create(
      programBuilder, programBuilder.getLoc(), val3.getType(), val3, val1);

  hState.propagateClassicalOperation(selectOp, val3, val0, val1, val2);
  hState.propagateClassicalOperation(subIOp, val3, val3, val1);

  const auto resStr = hState.toString();

  EXPECT_THAT(resStr, testing::AnyOf(testing::HasSubstr("integerValue0 = 0"),
                                     testing::HasSubstr("integerValue1 = 0"),
                                     testing::HasSubstr("integerValue2 = 0"),
                                     testing::HasSubstr("integerValue3 = 0")));
  EXPECT_THAT(resStr, testing::AnyOf(testing::HasSubstr("integerValue0 = 3"),
                                     testing::HasSubstr("integerValue1 = 3"),
                                     testing::HasSubstr("integerValue2 = 3"),
                                     testing::HasSubstr("integerValue3 = 3")));
  EXPECT_THAT(resStr, testing::AnyOf(testing::HasSubstr("integerValue0 = 9"),
                                     testing::HasSubstr("integerValue1 = 9"),
                                     testing::HasSubstr("integerValue2 = 9"),
                                     testing::HasSubstr("integerValue3 = 9")));
  EXPECT_THAT(resStr, testing::AnyOf(testing::HasSubstr("integerValue0 = 6"),
                                     testing::HasSubstr("integerValue1 = 6"),
                                     testing::HasSubstr("integerValue2 = 6"),
                                     testing::HasSubstr("integerValue3 = 6")));
}

TEST_F(HybridStateTest, doubleOpOneValueOperation) {
  auto hState = HybridState(fourQubits, 3);
  const auto i1 = programBuilder.getF64FloatAttr(-2.7);
  const auto i2 = programBuilder.getF64FloatAttr(0);

  const mlir::Value val1 = mlir::arith::ConstantOp::create(
      programBuilder, programBuilder.getLoc(), i1);
  const mlir::Value val2 = mlir::arith::ConstantOp::create(
      programBuilder, programBuilder.getLoc(), i2);

  hState.addDoubleValue(val1, -2.7);
  hState.addDoubleValue(val2, 0);

  const auto negFOp = mlir::arith::NegFOp::create(
      programBuilder, programBuilder.getLoc(), val2.getType(), val1);

  hState.propagateClassicalOperation(negFOp, val2, val1);

  const auto resStr = hState.toString();

  EXPECT_THAT(resStr,
              testing::AnyOf(testing::HasSubstr("doubleValue0 = -2.7"),
                             testing::HasSubstr("doubleValue1 = -2.7")));
  EXPECT_THAT(resStr, testing::AnyOf(testing::HasSubstr("doubleValue0 = 2.7"),
                                     testing::HasSubstr("doubleValue1 = 2.7")));
}

TEST_F(HybridStateTest, doubleOpTwoValueOperation) {
  auto hState = HybridState(fourQubits, 3);
  const auto i1 = programBuilder.getF64FloatAttr(-2.5);
  const auto i2 = programBuilder.getF64FloatAttr(1.3);

  const mlir::Value val1 = mlir::arith::ConstantOp::create(
      programBuilder, programBuilder.getLoc(), i1);
  const mlir::Value val2 = mlir::arith::ConstantOp::create(
      programBuilder, programBuilder.getLoc(), i2);

  hState.addDoubleValue(val1, -2.5);
  hState.addDoubleValue(val2, 1.3);

  const auto mulFOp = mlir::arith::MulFOp::create(
      programBuilder, programBuilder.getLoc(), val2.getType(), val1);

  hState.propagateClassicalOperation(mulFOp, val2, val1, val2);

  const auto resStr = hState.toString();

  EXPECT_THAT(resStr,
              testing::AnyOf(testing::HasSubstr("doubleValue0 = -2.50"),
                             testing::HasSubstr("doubleValue1 = -2.50")));
  EXPECT_THAT(resStr,
              testing::AnyOf(testing::HasSubstr("doubleValue0 = -3.25"),
                             testing::HasSubstr("doubleValue1 = -3.25")));
}
