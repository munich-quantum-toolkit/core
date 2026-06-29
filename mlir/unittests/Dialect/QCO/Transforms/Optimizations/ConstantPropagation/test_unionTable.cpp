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
#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/UnionTable.hpp"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <mlir/IR/Value.h>

#include <mlir/Dialect/Func/IR/FuncOpsDialect.h.inc>

using namespace mlir::qco;

class UnionTableTest : public testing::Test {
protected:
  mlir::MLIRContext context;
  QCOProgramBuilder programBuilder;
  UnionTable ut = UnionTable(4, 4);

  HOp hOp;
  XOp xOp;
  SWAPOp swapOp;

  mlir::Value v0;
  mlir::Value v1;
  mlir::Value v2;
  mlir::Value v3;
  mlir::Value v4;
  mlir::Value v5;
  mlir::Value v6;
  mlir::Value v7;
  mlir::Value v8;
  mlir::Value v9;
  mlir::Value i0;
  mlir::Value i1;
  mlir::Value i2;

  std::vector<mlir::Value> q0;
  std::vector<mlir::Value> q1;
  std::vector<mlir::Value> q2;
  std::vector<mlir::Value> q3;
  std::vector<mlir::Value> q4;
  std::vector<mlir::Value> q5;
  std::vector<mlir::Value> q6;
  std::vector<mlir::Value> q7;
  std::vector<mlir::Value> q8;
  std::vector<mlir::Value> q9;

  UnionTableTest() : programBuilder(&context) {}

  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<QCODialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    programBuilder.initialize();

    auto q = programBuilder.allocQubitRegister(10);
    hOp = HOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    xOp = XOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    swapOp = SWAPOp::create(programBuilder, programBuilder.getLoc(),
                            {q[0].getType(), q[1].getType()}, {q[0], q[1]});

    v0 = q[0];
    v1 = q[1];
    v2 = q[2];
    v3 = q[3];
    v4 = q[4];
    v5 = q[5];
    v6 = q[6];
    v7 = q[7];
    v8 = q[8];
    v9 = q[9];

    q0 = {v0};
    q1 = {v1};
    q2 = {v2};
    q3 = {v3};
    q4 = {v4};
    q5 = {v5};
    q6 = {v6};
    q7 = {v7};
    q8 = {v8};
    q9 = {v9};

    const auto iAttr = programBuilder.getI64IntegerAttr(0);

    i0 = mlir::arith::ConstantOp::create(programBuilder,
                                         programBuilder.getLoc(), iAttr);
    i1 = mlir::arith::ConstantOp::create(programBuilder,
                                         programBuilder.getLoc(), iAttr);
    i2 = mlir::arith::ConstantOp::create(programBuilder,
                                         programBuilder.getLoc(), iAttr);

    ut.propagateQubitAlloc(v0);
    ut.propagateQubitAlloc(v1);
    ut.propagateQubitAlloc(v2);
    ut.propagateQubitAlloc(v3);
  }

  void TearDown() override {}
};

TEST_F(UnionTableTest, ApplyHGate) {
  ut.propagateGate(hOp, q0, q5);

  EXPECT_THAT(
      ut.toString(),
      testing::HasSubstr(
          "Qubits: 0, HybridStates: {{|0> -> 0.71, |1> -> 0.71}: p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyHGateToThirdQubit) {
  ut.propagateGate(hOp, q2, q5);

  EXPECT_THAT(
      ut.toString(),
      testing::HasSubstr(
          "Qubits: 2, HybridStates: {{|0> -> 0.71, |1> -> 0.71}: p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyQuantumControlledGate) {
  ut.propagateGate(hOp, q1, q5);
  ut.propagateGate(xOp, q3, q6);
  ut.propagateGate(xOp, q6, q7, q5);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 0, HybridStates: {{|0> -> 1.00}: p = 1.00;}"));
  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 2, HybridStates: {{|0> -> 1.00}: p = 1.00;}"));
  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 31, HybridStates: {{|01> -> 0.71, "
                                 "|10> -> 0.71}: p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyClassicalControlledGateThatsFalse) {
  std::vector classicalControl = {i1};
  ut.propagateIntAlloc(i0, 1);
  ut.propagateIntAlloc(i1, 0);
  ut.propagateGate(hOp, q1, q5);
  ut.propagateGate(xOp, q3, q6);
  ut.propagateGate(xOp, q6, q7, q5, classicalControl);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 31, HybridStates: {{|10> "
                  "-> 0.71, |11> -> 0.71}: integerValue0 = 0; p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyClassicalControlledGateThatsTrue) {
  std::vector classicalControl = {i0};
  ut.propagateIntAlloc(i0, 1);
  ut.propagateIntAlloc(i1, 0);
  ut.propagateGate(hOp, q1, q5);
  ut.propagateGate(xOp, q3, q6);
  ut.propagateGate(xOp, q6, q7, q5, classicalControl);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 31, HybridStates: {{|01> "
                  "-> 0.71, |10> -> 0.71}: integerValue0 = 1; p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyNegClassicalControlledGateThatsTrue) {
  std::vector classicalControl = {i0};
  ut.propagateIntAlloc(i0, 1);
  ut.propagateIntAlloc(i1, 0);
  ut.propagateGate(hOp, q1, q5);
  ut.propagateGate(xOp, q3, q6);
  ut.propagateGate(xOp, q6, q7, q5, {}, classicalControl);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 31, HybridStates: {{|10> "
                  "-> 0.71, |11> -> 0.71}: integerValue0 = 1; p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyNegClassicalControlledGateThatsFalse) {
  std::vector classicalControl = {i1};
  ut.propagateIntAlloc(i0, 1);
  ut.propagateIntAlloc(i1, 0);
  ut.propagateGate(hOp, q1, q5);
  ut.propagateGate(xOp, q3, q6);
  ut.propagateGate(xOp, q6, q7, q5, {}, classicalControl);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 31, HybridStates: {{|01> "
                  "-> 0.71, |10> -> 0.71}: integerValue0 = 0; p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyPosNegClassicalControlledGateThatsFalse) {
  std::vector classicalControlZero = {i0};
  std::vector classicalControlOne = {i1};
  ut.propagateIntAlloc(i0, 0);
  ut.propagateIntAlloc(i1, 0);
  ut.propagateGate(hOp, q1, q5);
  ut.propagateGate(xOp, q3, q6);
  ut.propagateGate(xOp, q6, q7, q5, classicalControlOne, classicalControlZero);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 31, HybridStates: {{|10> "
                                 "-> 0.71, |11> -> 0.71}: integerValue0 = 0, "
                                 "integerValue1 = 0; p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyPosNegClassicalControlledGateThatsTrue) {
  std::vector classicalControlTrue = {i0};
  std::vector classicalControlFalse = {i1};
  ut.propagateIntAlloc(i0, 1);
  ut.propagateIntAlloc(i1, 0);
  ut.propagateGate(hOp, q1, q5);
  ut.propagateGate(xOp, q3, q6);
  ut.propagateGate(xOp, q6, q7, q5, classicalControlTrue,
                   classicalControlFalse);

  EXPECT_THAT(
      ut.toString(),
      testing::AnyOf(
          testing::HasSubstr("Qubits: 31, HybridStates: {{|01> "
                             "-> 0.71, |10> -> 0.71}: integerValue0 = 1, "
                             "integerValue1 = 0; p = 1.00;}"),
          testing::HasSubstr("Qubits: 31, HybridStates: {{|01> "
                             "-> 0.71, |10> -> 0.71}: integerValue0 = 0, "
                             "integerValue1 = 1; p = 1.00;}")));
}

TEST_F(UnionTableTest, ApplyControlledTwoBitGate) {
  std::vector classicalControlTrue = {i0, i2};
  std::vector classicalControlFalse = {i1};
  ut.propagateIntAlloc(i0, 1);
  ut.propagateIntAlloc(i1, 0);
  ut.propagateIntAlloc(i2, 1);
  ut.propagateGate(hOp, q1, q4);
  ut.propagateGate(xOp, q3, q5);
  ut.propagateGate(xOp, q5, q6, q4, classicalControlTrue,
                   classicalControlFalse);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 31, HybridStates: {{|01> "
                                 "-> 0.71, |10> -> 0.71}: "));
}

TEST_F(UnionTableTest, doMeasurementWithOneResult) {
  ut.propagateGate(xOp, q0, q4);
  ut.propagateIntAlloc(i0, 10);
  ut.propagateMeasurement(v4, v5, i0);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 0, HybridStates: {{|1> "
                                 "-> 1.00}: integerValue0 = 1; p = 1.00;}"));
}

TEST_F(UnionTableTest, doMeasurementWithTwoResults) {
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateIntAlloc(i0, 10);
  ut.propagateMeasurement(v5, v6, i0);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 10, HybridStates: {{|00> "
                                 "-> 1.00}: integerValue0 = 0; p = 0.50; {|11> "
                                 "-> 1.00}: integerValue0 = 1; p = 0.50;}"));
}

TEST_F(UnionTableTest, doMeasurementWithNegPosCtrl) {
  std::vector ctrl = {i0};
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateIntAlloc(i0, 0);
  ut.propagateMeasurement(v5, v6, i0, ctrl);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 10, HybridStates: {{|00> "
                  "-> 0.71, |11> -> 0.71}: integerValue0 = 0; p = 1.00;}"));
}

TEST_F(UnionTableTest, doMeasurementWithPosNegCtrl) {
  std::vector ctrl = {i0};
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateIntAlloc(i0, 10);
  ut.propagateMeasurement(v5, v6, i0, {}, ctrl);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 10, HybridStates: {{|00> "
                  "-> 0.71, |11> -> 0.71}: integerValue0 = 10; p = 1.00;}"));
}

TEST_F(UnionTableTest, doResetWithOneResult) {
  ut.propagateGate(xOp, q0, q4);
  ut.propagateReset(v4, v5);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 0, HybridStates: {{|0> "
                                 "-> 1.00}: p = 1.00;}"));
}

TEST_F(UnionTableTest, doResetWithTwoResults) {
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateReset(v4, v6);

  EXPECT_THAT(
      ut.toString(),
      testing::HasSubstr("Qubits: 10, HybridStates: {{|00> "
                         "-> 1.00}: p = 0.50; {|10> -> 1.00}: p = 0.50;}"));
}

TEST_F(UnionTableTest, swapGateApplicationDifferentStates) {
  std::vector swapTargets = {v6, v2};
  std::vector swapDestinations = {v7, v8};
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateGate(xOp, q5, q6);
  ut.propagateGate(swapOp, swapTargets, swapDestinations);
  ut.propagateGate(xOp, q7, q9);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 20, HybridStates: {{|01> -> 0.71, "
                                 "|10> -> 0.71}: p = 1.00;}"));
  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 1, HybridStates: {{|1> -> 1.00}: p = 1.00;}"));
}

TEST_F(UnionTableTest, swapGateApplicationSameState) {
  std::vector swapTargets = {v4, v6};
  std::vector swapDestinations = {v7, v8};
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(hOp, q1, q5, q4);
  ut.propagateGate(xOp, q5, q6);
  ut.propagateGate(swapOp, swapTargets, swapDestinations);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 10, HybridStates: {{|01> -> 0.71, "
                                 "|10> -> 0.50, |11> -> 0.50}: p = 1.00;}"));
}

class UnionTableWithoutSetupAllocationsTest : public testing::Test {
protected:
  mlir::MLIRContext context;
  QCOProgramBuilder programBuilder;
  QCOProgramBuilder referenceBuilder;

  HOp hOp;
  XOp xOp;

  mlir::Value v0;
  mlir::Value v1;
  mlir::Value v2;
  mlir::Value v3;
  mlir::Value v4;
  mlir::Value v5;
  mlir::Value i0;

  std::vector<mlir::Value> q0;
  std::vector<mlir::Value> q1;
  std::vector<mlir::Value> q2;
  std::vector<mlir::Value> q3;
  std::vector<mlir::Value> q4;
  std::vector<mlir::Value> q5;

  UnionTableWithoutSetupAllocationsTest()
      : programBuilder(&context), referenceBuilder(&context) {}

  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<QCODialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    programBuilder.initialize();
    referenceBuilder.initialize();

    auto q = programBuilder.allocQubitRegister(6);
    hOp = HOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    xOp = XOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    v0 = q[0];
    v1 = q[1];
    v2 = q[2];
    v3 = q[3];
    v4 = q[4];
    v5 = q[5];

    q0 = {v0};
    q1 = {v1};
    q2 = {v2};
    q3 = {v3};
    q4 = {v4};
    q5 = {v5};

    const auto iAttr = programBuilder.getI64IntegerAttr(0);

    i0 = mlir::arith::ConstantOp::create(programBuilder,
                                         programBuilder.getLoc(), iAttr);
  }

  void TearDown() override {}
};

TEST_F(UnionTableWithoutSetupAllocationsTest, propagateQubitAlloc) {
  auto ut = UnionTable(4, 2);
  ut.propagateQubitAlloc(v0);
  ut.propagateQubitAlloc(v1);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 0, HybridStates: {{|0> -> 1.00}: p = 1.00;}"));
  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 1, HybridStates: {{|0> -> 1.00}: p = 1.00;}"));
}

TEST_F(UnionTableWithoutSetupAllocationsTest, doMeasurementsAndGetToTop) {
  auto ut = UnionTable(4, 1);
  ut.propagateQubitAlloc(v1);
  ut.propagateQubitAlloc(v2);
  ut.propagateIntAlloc(i0, 10);
  ut.propagateGate(hOp, q1, q3);
  ut.propagateMeasurement(v3, v4, i0);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 0, HybridStates: {TOP}"));
}

TEST_F(UnionTableWithoutSetupAllocationsTest, doMeasurementsOnTop) {
  auto ut = UnionTable(2, 2);
  ut.propagateQubitAlloc(v0);
  ut.propagateQubitAlloc(v1);
  ut.propagateIntAlloc(i0, 10);
  ut.propagateGate(hOp, q0, q2);
  ut.propagateGate(xOp, q1, q3, q2);
  ut.propagateGate(hOp, q3, q4); // State enters TOP
  ut.propagateMeasurement(v4, v5, i0);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 10, HybridStates: {TOP}"));
}

TEST_F(UnionTableWithoutSetupAllocationsTest, doResetOnTop) {
  auto ut = UnionTable(2, 2);
  ut.propagateQubitAlloc(v0);
  ut.propagateQubitAlloc(v1);
  ut.propagateGate(hOp, q0, q2);
  ut.propagateGate(xOp, q1, q3, q2);
  ut.propagateGate(hOp, q3, q4); // State enters TOP
  ut.propagateReset(v4, v5);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 10, HybridStates: {TOP}"));
}

TEST_F(UnionTableWithoutSetupAllocationsTest, unifyTooLargeHybridStates) {
  auto ut = UnionTable(4, 1);
  ut.propagateQubitAlloc(v0);
  ut.propagateQubitAlloc(v1);
  ut.propagateQubitAlloc(v2);
  ut.propagateIntAlloc(i0, 10);
  ut.propagateGate(hOp, q0, q3);
  ut.propagateMeasurement(v3, v4, i0);
  ut.propagateGate(xOp, q1, q5, q4);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 10, HybridStates: {TOP}"));
}

class UnionTablePropertiesTest : public testing::Test {
protected:
  mlir::MLIRContext context;
  QCOProgramBuilder programBuilder;
  UnionTable ut = UnionTable(3, 2);

  HOp hOp;
  XOp xOp;
  ZOp zOp;
  SWAPOp swapOp;

  mlir::Value v0;
  mlir::Value v1;
  mlir::Value v2;
  mlir::Value v3;
  mlir::Value v4;
  mlir::Value v5;
  mlir::Value v6;
  mlir::Value v7;
  mlir::Value v8;
  mlir::Value v9;
  mlir::Value v10;
  mlir::Value i0;
  mlir::Value i1;

  std::vector<mlir::Value> q0;
  std::vector<mlir::Value> q1;
  std::vector<mlir::Value> q2;
  std::vector<mlir::Value> q3;
  std::vector<mlir::Value> q4;
  std::vector<mlir::Value> q5;
  std::vector<mlir::Value> q6;
  std::vector<mlir::Value> q7;
  std::vector<mlir::Value> q8;
  std::vector<mlir::Value> q9;
  std::vector<mlir::Value> q10;

  UnionTablePropertiesTest() : programBuilder(&context) {}

  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<QCODialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    programBuilder.initialize();

    auto q = programBuilder.allocQubitRegister(11);
    hOp = HOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    xOp = XOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    zOp = ZOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    swapOp = SWAPOp::create(programBuilder, programBuilder.getLoc(),
                            {q[0].getType(), q[1].getType()}, {q[0], q[1]});

    v0 = q[0];
    v1 = q[1];
    v2 = q[2];
    v3 = q[3];
    v4 = q[4];
    v5 = q[5];
    v6 = q[6];
    v7 = q[7];
    v8 = q[8];
    v9 = q[9];
    v10 = q[10];

    q0 = {v0};
    q1 = {v1};
    q2 = {v2};
    q3 = {v3};
    q4 = {v4};
    q5 = {v5};
    q6 = {v6};
    q7 = {v7};
    q8 = {v8};
    q9 = {v9};
    q10 = {v10};

    const auto iAttr = programBuilder.getI64IntegerAttr(0);

    i0 = mlir::arith::ConstantOp::create(programBuilder,
                                         programBuilder.getLoc(), iAttr);
    i1 = mlir::arith::ConstantOp::create(programBuilder,
                                         programBuilder.getLoc(), iAttr);

    ut.propagateQubitAlloc(v0);
    ut.propagateQubitAlloc(v1);
    ut.propagateQubitAlloc(v2);
    ut.propagateIntAlloc(i0, 0);
  }

  void TearDown() override {}
};

TEST_F(UnionTablePropertiesTest, alwaysZeroOneAreFalse) {
  std::vector ctrl = {i0};
  ut.propagateGate(hOp, q0, q3);
  ut.propagateGate(xOp, q1, q4, q3);
  ut.propagateGate(xOp, q2, q5, q4);
  ut.propagateGate(xOp, q3, q6);
  ut.propagateMeasurement(v4, v7, i0);
  ut.propagateGate(hOp, q6, q8, {}, ctrl);

  EXPECT_FALSE(ut.isQubitAlwaysZero(v5));
  EXPECT_FALSE(ut.isQubitAlwaysOne(v8));
}

TEST_F(UnionTablePropertiesTest, alwaysZeroIsTrue) {
  std::vector ctrl = {i0};
  ut.propagateGate(hOp, q0, q3);
  ut.propagateGate(xOp, q1, q4, q3);
  ut.propagateGate(xOp, q2, q5, q4);
  ut.propagateGate(xOp, q3, q6);
  ut.propagateMeasurement(v4, v7, i0);
  ut.propagateGate(xOp, q5, q8, {}, ctrl);
  ut.propagateGate(hOp, q6, q9, {}, ctrl);

  EXPECT_TRUE(ut.isQubitAlwaysZero(v8));
}

TEST_F(UnionTablePropertiesTest, alwaysOneIsTrue) {
  std::vector ctrl = {i0};
  std::vector qCtrl = {v7, v4};
  ut.propagateGate(hOp, q0, q3);
  ut.propagateGate(xOp, q1, q4, q3);
  ut.propagateGate(xOp, q2, q5, q4);
  ut.propagateGate(xOp, q3, q6);
  ut.propagateMeasurement(v5, v7, i0);
  ut.propagateGate(hOp, q6, q8, {}, ctrl);
  ut.propagateGate(zOp, q8, q9, qCtrl);
  ut.propagateGate(hOp, q9, q10, {}, ctrl);

  EXPECT_TRUE(ut.isQubitAlwaysOne(v10));
}

TEST_F(UnionTablePropertiesTest, bitAlwaysZeroIsTrueOneIsFalse) {
  std::vector ctrl = {i0};
  ut.propagateIntAlloc(i1, 0);
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateMeasurement(v4, v6, i0);
  ut.propagateGate(xOp, q5, q7, {}, ctrl);
  ut.propagateMeasurement(v7, v8, i1);

  EXPECT_FALSE(ut.isClassicalValueAlwaysTrue(i0));
  EXPECT_TRUE(ut.isClassicalValueAlwaysFalse(i1));
}

TEST_F(UnionTablePropertiesTest, bitAlwaysZeroIsFalseOneIsTrue) {
  std::vector ctrl = {i0};
  ut.propagateIntAlloc(i1, 0);
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateMeasurement(v4, v6, i0);
  ut.propagateGate(xOp, q5, q7, {}, {}, ctrl);
  ut.propagateMeasurement(v7, v8, i1);

  EXPECT_TRUE(ut.isClassicalValueAlwaysTrue(i1));
  EXPECT_FALSE(ut.isClassicalValueAlwaysFalse(i0));
}

TEST_F(UnionTablePropertiesTest, testAllTopAmplitudes) {
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateGate(xOp, q2, q6, q5);
  ut.propagateMeasurement(v6, v7, i0);
  ut.propagateGate(hOp, q4, q8);
  EXPECT_FALSE(ut.areStatesAllTop());
  ut.propagateGate(hOp, q5, q9);
  EXPECT_TRUE(ut.areStatesAllTop());
  ut.propagateMeasurement(v8, v10, i0);
  EXPECT_TRUE(ut.areStatesAllTop());
}

TEST_F(UnionTablePropertiesTest, testAllTopHybridStates) {
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateGate(xOp, q2, q6, q5);
  ut.propagateMeasurement(v6, v7, i0);
  ut.propagateGate(hOp, q4, q8);
  EXPECT_FALSE(ut.areStatesAllTop());
  ut.propagateMeasurement(v8, v10, i0);
  EXPECT_TRUE(ut.areStatesAllTop());
}

TEST_F(UnionTablePropertiesTest, testMinusOneGlobalPhase) {
  std::vector qCtrl = {v3, v4};
  ut.propagateGate(xOp, q0, q3);
  ut.propagateGate(xOp, q1, q4);
  ut.propagateGate(xOp, q2, q5);
  const auto globalPhase = ut.globalPhaseThatIsAdded(zOp, v5, qCtrl);
  EXPECT_TRUE(globalPhase.has_value());
  EXPECT_EQ(std::numbers::pi, globalPhase.value());
}

TEST_F(UnionTablePropertiesTest, testOneGlobalPhase) {
  std::vector qCtrl = {v4, v5};
  ut.propagateGate(xOp, q0, q4);
  ut.propagateGate(hOp, q1, q5);
  const auto emptyGlobalPhase = ut.globalPhaseThatIsAdded(zOp, v5, q4);
  const auto globalPhase = ut.globalPhaseThatIsAdded(zOp, v2, qCtrl);
  EXPECT_FALSE(emptyGlobalPhase.has_value());
  EXPECT_TRUE(globalPhase.has_value());
  EXPECT_EQ(0.0, globalPhase.value());
}

TEST_F(UnionTablePropertiesTest, FindEquivalentClassicalValue) {
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateMeasurement(v5, v6, i0);
  const llvm::DenseMap<mlir::Value, bool> result =
      ut.getValueThatIsEquivalentToQubit(v6);
  ASSERT_FALSE(result.empty());
  ASSERT_TRUE(result.at(i0));
}

TEST_F(UnionTablePropertiesTest, FindEquivalentReversedClassicalValue) {
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateGate(xOp, q4, q6);
  ut.propagateMeasurement(v5, v7, i0);
  const llvm::DenseMap<mlir::Value, bool> result =
      ut.getValueThatIsEquivalentToQubit(v6);
  ASSERT_FALSE(result.empty());
  ASSERT_FALSE(result.at(i0));
}

TEST_F(UnionTablePropertiesTest, FindNoEquivalentClassicalValue) {
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateGate(hOp, q4, q6);
  ut.propagateMeasurement(v5, v7, i0);
  const llvm::DenseMap<mlir::Value, bool> result =
      ut.getValueThatIsEquivalentToQubit(v6);
  ASSERT_TRUE(result.empty());
}

TEST_F(UnionTablePropertiesTest, hasAlwaysZeroProbabilityTest) {
  std::vector classicalIndexVec = {i0};
  ut.propagateIntAlloc(i1, 20);
  ut.propagateGate(hOp, q0, q3);
  ut.propagateGate(xOp, q1, q4, q3);
  ut.propagateMeasurement(v4, v5, i0);

  llvm::DenseMap<mlir::Value, bool> qubits0;
  llvm::DenseMap<mlir::Value, bool> classicals0;
  llvm::DenseMap<mlir::Value, bool> qubits1;
  llvm::DenseMap<mlir::Value, bool> classicals1;
  qubits0[v3] = true;
  qubits0[v5] = true;
  qubits0[v2] = false;
  classicals0[i0] = true;
  classicals0[i1] = true;
  qubits1[v3] = false;
  qubits1[v5] = false;
  qubits1[v2] = true;
  classicals1[i1] = false;

  ASSERT_FALSE(ut.hasAlwaysZeroProbability(qubits0, classicals0));
  ASSERT_TRUE(ut.hasAlwaysZeroProbability(qubits1, classicals1));
}

TEST_F(UnionTablePropertiesTest, ZeroIsAlwaysAntecedent) {
  std::vector classicalIndexVec = {i0};
  ut.propagateMeasurement(v0, v4, i0);
  ut.propagateGate(hOp, q1, q5);
  ASSERT_TRUE(ut.isQubitImplied(v5, q4, classicalIndexVec, {}));
}

TEST_F(UnionTablePropertiesTest, ImpliedQubit) {
  std::vector classicalIndexVec = {i0};
  ut.propagateGate(hOp, q0, q4);
  ut.propagateMeasurement(v4, v5, i0);
  ut.propagateGate(hOp, q1, q6, q5);
  ASSERT_TRUE(ut.isQubitImplied(v5, q6, classicalIndexVec, {}));
  ASSERT_FALSE(ut.isQubitImplied(v6, q5, classicalIndexVec, {}));
}

TEST_F(UnionTablePropertiesTest, ImpliedQubitOnlyQubits) {
  ut.propagateGate(hOp, q0, q4);
  ut.propagateMeasurement(v4, v5, i0);
  ut.propagateGate(hOp, q1, q6, q5);
  ASSERT_TRUE(ut.isQubitImplied(v5, q6, {}, {}));
}

TEST_F(UnionTablePropertiesTest, ImpliedQubitOnlyClassicalValues) {
  std::vector classicalIndexVec = {i0};
  ut.propagateGate(hOp, q0, q4);
  ut.propagateMeasurement(v4, v5, i0);
  ut.propagateGate(hOp, q5, q6, {}, {}, classicalIndexVec);
  ASSERT_TRUE(ut.isQubitImplied(v6, {}, classicalIndexVec, {}));
}

TEST_F(UnionTablePropertiesTest, ImpliedQubitNegClassicalValues) {
  std::vector classicalIndexVec = {i0};
  ut.propagateGate(hOp, q0, q4);
  ut.propagateMeasurement(v4, v5, i0);
  ut.propagateGate(hOp, q5, q6, {}, classicalIndexVec);
  ut.propagateGate(xOp, q6, q7);
  ASSERT_TRUE(ut.isQubitImplied(v7, {}, {}, classicalIndexVec));
}

TEST_F(UnionTablePropertiesTest, globalPhaseOneQubit) {
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5);

  const auto globalPhase0 = ut.globalPhaseThatIsAdded(zOp, v4);
  const auto globalPhase1 = ut.globalPhaseThatIsAdded(zOp, v5);

  ASSERT_FALSE(globalPhase0.has_value());
  ASSERT_TRUE(globalPhase1.has_value());
  ASSERT_EQ(std::numbers::pi, globalPhase1.value());
}

TEST_F(UnionTablePropertiesTest, noGlobalPhaseTwoQubitsA) {
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(hOp, q1, q5);

  const auto globalPhase0 = ut.globalPhaseThatIsAdded(zOp, v4, q5);

  ASSERT_FALSE(globalPhase0.has_value());
}

TEST_F(UnionTablePropertiesTest, noGlobalPhaseTwoQubitsB) {
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(hOp, q1, q5, q4);
  ut.propagateMeasurement(v5, v6, i0);

  const auto globalPhase0 = ut.globalPhaseThatIsAdded(zOp, v4, q6);

  ASSERT_FALSE(globalPhase0.has_value());
}

TEST_F(UnionTablePropertiesTest, globalPhaseTwoQubits) {
  ut.propagateQubitAlloc(v3);
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q2, q5);
  ut.propagateGate(xOp, q3, q6);

  const auto globalPhase0 = ut.globalPhaseThatIsAdded(zOp, v1, q4);
  const auto globalPhase1 = ut.globalPhaseThatIsAdded(zOp, v5, q6);

  ASSERT_TRUE(globalPhase0.has_value());
  ASSERT_TRUE(globalPhase1.has_value());
  ASSERT_EQ(0.0, globalPhase0);
  ASSERT_EQ(std::numbers::pi, globalPhase1);
}

TEST_F(UnionTablePropertiesTest, findNonSatisfiableCombinationsA) {
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5);
  ut.propagateGate(xOp, q5, q6, q4);
  std::vector combinations = {v4, v6};

  ASSERT_FALSE(ut.areThereSatisfiableCombinations(combinations));
}

TEST_F(UnionTablePropertiesTest, findNonSatisfiableCombinationsB) {
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  std::vector combinations = {v4, v5};

  ASSERT_TRUE(ut.areThereSatisfiableCombinations(combinations));
}

TEST_F(UnionTablePropertiesTest, findNonSatisfiableCombinationsC) {
  ut.propagateIntAlloc(i1, 2);
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateMeasurement(v4, v6, i0);
  ut.propagateMeasurement(v5, v7, i1);
  ut.propagateGate(hOp, q6, q8);
  ut.propagateGate(hOp, q7, q9, q8);

  std::vector qubitCombinations = {v8, v9};
  std::vector classicalCombinations = {i0, i1};
  std::vector classicalVal0 = {i0};
  std::vector classicalVal1 = {i1};

  ASSERT_TRUE(ut.areThereSatisfiableCombinations(qubitCombinations,
                                                 classicalCombinations));
  ASSERT_FALSE(
      ut.areThereSatisfiableCombinations({}, classicalVal0, classicalVal1));
  ASSERT_TRUE(ut.areThereSatisfiableCombinations(qubitCombinations, {},
                                                 classicalCombinations));
}

class SmallUnionTableTest : public testing::Test {
protected:
  mlir::MLIRContext context;
  QCOProgramBuilder programBuilder;
  UnionTable ut = UnionTable(2, 2);

  HOp hOp;
  XOp xOp;

  mlir::Value v0;
  mlir::Value v1;
  mlir::Value v2;
  mlir::Value v3;
  mlir::Value v4;
  mlir::Value v5;
  mlir::Value v6;
  mlir::Value v7;

  std::vector<mlir::Value> q0;
  std::vector<mlir::Value> q1;
  std::vector<mlir::Value> q2;
  std::vector<mlir::Value> q3;
  std::vector<mlir::Value> q4;
  std::vector<mlir::Value> q5;
  std::vector<mlir::Value> q6;
  std::vector<mlir::Value> q7;

  SmallUnionTableTest() : programBuilder(&context) {}

  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<QCODialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    programBuilder.initialize();

    auto q = programBuilder.allocQubitRegister(8);
    hOp = HOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    xOp = XOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);

    v0 = q[0];
    v1 = q[1];
    v2 = q[2];
    v3 = q[3];
    v4 = q[4];
    v5 = q[5];
    v6 = q[6];
    v7 = q[7];

    q0 = {v0};
    q1 = {v1};
    q2 = {v2};
    q3 = {v3};
    q4 = {v4};
    q5 = {v5};
    q6 = {v6};
    q7 = {v7};

    ut.propagateQubitAlloc(v0);
    ut.propagateQubitAlloc(v1);
    ut.propagateQubitAlloc(v2);
    ut.propagateQubitAlloc(v3);
  }
};

TEST_F(SmallUnionTableTest, handleErrorIfTwoManyAmplitudesAreNonzero) {
  ut.propagateGate(hOp, q3, q4);
  ut.propagateGate(xOp, q2, q5, q4);
  ut.propagateGate(hOp, q5, q6);
  ut.propagateGate(hOp, q6, q7);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 32, HybridStates: {TOP}"));
}

TEST_F(SmallUnionTableTest, applyGatesOnPartiallyTopQState) {
  ut.propagateGate(hOp, q2, q4);
  ut.propagateGate(hOp, q3, q5);
  ut.propagateGate(xOp, q4, q6, q5); // Qubit 2 and 3 enter TOP
  ut.propagateGate(xOp, q1, q7, q6);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 321, HybridStates: {TOP}"));
}

class UnionTableSuperfluousTest : public testing::Test {
protected:
  mlir::MLIRContext context;
  QCOProgramBuilder programBuilder;
  UnionTable ut = UnionTable(8, 4);

  HOp hOp;
  XOp xOp;
  ZOp zOp;

  mlir::Value v0;
  mlir::Value v1;
  mlir::Value v2;
  mlir::Value v3;
  mlir::Value v4;
  mlir::Value v5;
  mlir::Value v6;
  mlir::Value v7;
  mlir::Value v8;
  mlir::Value v9;
  mlir::Value v10;
  mlir::Value v11;
  mlir::Value v12;
  mlir::Value v13;
  mlir::Value v14;
  mlir::Value v15;
  mlir::Value v16;
  mlir::Value v17;
  mlir::Value v18;
  mlir::Value i0;
  mlir::Value i1;
  mlir::Value i2;
  mlir::Value i3;

  std::vector<mlir::Value> q0;
  std::vector<mlir::Value> q1;
  std::vector<mlir::Value> q2;
  std::vector<mlir::Value> q3;
  std::vector<mlir::Value> q4;
  std::vector<mlir::Value> q5;
  std::vector<mlir::Value> q6;
  std::vector<mlir::Value> q7;
  std::vector<mlir::Value> q8;
  std::vector<mlir::Value> q9;
  std::vector<mlir::Value> q10;
  std::vector<mlir::Value> q11;
  std::vector<mlir::Value> q12;
  std::vector<mlir::Value> q13;
  std::vector<mlir::Value> q14;
  std::vector<mlir::Value> q15;
  std::vector<mlir::Value> q16;
  std::vector<mlir::Value> q17;
  std::vector<mlir::Value> q18;

  UnionTableSuperfluousTest() : programBuilder(&context) {}

  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<QCODialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    programBuilder.initialize();

    auto q = programBuilder.allocQubitRegister(17);
    hOp = HOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    xOp = XOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);
    zOp = ZOp::create(programBuilder, programBuilder.getLoc(), q[0].getType(),
                      q[0]);

    v0 = q[0];
    v1 = q[1];
    v2 = q[2];
    v3 = q[3];
    v4 = q[4];
    v5 = q[5];
    v6 = q[6];
    v7 = q[7];
    v8 = q[8];
    v9 = q[9];
    v10 = q[10];
    v11 = q[11];
    v12 = q[12];
    v13 = q[13];
    v14 = q[14];
    v15 = q[15];
    v16 = q[16];

    q0 = {v0};
    q1 = {v1};
    q2 = {v2};
    q3 = {v3};
    q4 = {v4};
    q5 = {v5};
    q6 = {v6};
    q7 = {v7};
    q8 = {v8};
    q9 = {v9};
    q10 = {v10};
    q11 = {v11};
    q12 = {v12};
    q13 = {v13};
    q14 = {v14};
    q15 = {v15};
    q16 = {v16};

    const auto iAttr = programBuilder.getI64IntegerAttr(0);

    i0 = mlir::arith::ConstantOp::create(programBuilder,
                                         programBuilder.getLoc(), iAttr);
    i1 = mlir::arith::ConstantOp::create(programBuilder,
                                         programBuilder.getLoc(), iAttr);
    i2 = mlir::arith::ConstantOp::create(programBuilder,
                                         programBuilder.getLoc(), iAttr);
    i3 = mlir::arith::ConstantOp::create(programBuilder,
                                         programBuilder.getLoc(), iAttr);

    ut.propagateQubitAlloc(v0);
    ut.propagateQubitAlloc(v1);
    ut.propagateQubitAlloc(v2);
    ut.propagateQubitAlloc(v3);
    ut.propagateIntAlloc(i0, 10);
    ut.propagateIntAlloc(i1, 10);
    ut.propagateIntAlloc(i2, 10);
    ut.propagateIntAlloc(i3, 10);
    ut.propagateGate(hOp, q0, q4);
    ut.propagateMeasurement(v4, v5, i0);
    ut.propagateGate(hOp, q5, q6);
    ut.propagateMeasurement(v6, v7, i1);
    ut.propagateGate(hOp, q7, q8);

    ut.propagateGate(hOp, q1, q9);
    ut.propagateGate(hOp, q9, q10);
    ut.propagateMeasurement(v10, v11, i2); // classical value 2 = false

    ut.propagateGate(hOp, q2, q12);

    ut.propagateGate(hOp, q3, q13);
    ut.propagateGate(zOp, q13, q14);
    ut.propagateGate(hOp, q14, q15);
    ut.propagateMeasurement(v15, v16, i3); // classical value 3 = true
  }

  void TearDown() override {}
};

TEST_F(UnionTableSuperfluousTest, oneSuperfluousEach) {
  std::vector quantumCtrl = {v12, v16};
  std::vector posClassicalCtrl = {i0, i3};
  std::vector negClassicalCtrl = {i1, i2};
  auto [completelySuperfluous, superfluousQubits, superfluousClassicalValues] =
      ut.getSuperfluousControls(quantumCtrl, posClassicalCtrl,
                                negClassicalCtrl);
  ASSERT_EQ(superfluousQubits.size(), 1);
  ASSERT_EQ(superfluousClassicalValues.size(), 2);
  ASSERT_TRUE(superfluousQubits.contains(v16));
  ASSERT_TRUE(superfluousClassicalValues.contains(i2));
  ASSERT_TRUE(superfluousClassicalValues.contains(i3));
  ASSERT_FALSE(completelySuperfluous);
}

TEST_F(UnionTableSuperfluousTest, completelySuperfluousDueToNegQuantumCtrl) {
  std::vector quantumCtrl = {v11, v12, v16};
  std::vector posClassicalCtrl = {i0, i3};
  std::vector negClassicalCtrl = {i1, i2};
  const auto results = ut.getSuperfluousControls(quantumCtrl, posClassicalCtrl,
                                                 negClassicalCtrl);
  ASSERT_TRUE(results.completelySuperfluous);
}

TEST_F(UnionTableSuperfluousTest, completelySuperfluousDueToNegClassicalCtrl) {
  std::vector quantumCtrl = {v12, v16};
  std::vector posClassicalCtrl = {i0, i2, i3};
  std::vector negClassicalCtrl = {i1};
  const auto results = ut.getSuperfluousControls(quantumCtrl, posClassicalCtrl,
                                                 negClassicalCtrl);
  ASSERT_TRUE(results.completelySuperfluous);
}

TEST_F(UnionTableSuperfluousTest, completelySuperfluousDueToPosClassicalCtrl) {
  std::vector quantumCtrl = {v12, v16};
  std::vector posClassicalCtrl = {i0};
  std::vector negClassicalCtrl = {i1, i2, i3};
  const auto results = ut.getSuperfluousControls(quantumCtrl, posClassicalCtrl,
                                                 negClassicalCtrl);
  ASSERT_TRUE(results.completelySuperfluous);
}
