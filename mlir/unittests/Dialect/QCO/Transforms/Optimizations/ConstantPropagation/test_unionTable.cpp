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
                            {q[0].getType()}, {q[0], q[1], q[2], q[3]});

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
  ut.propagateGate(xOp, q6, q7, {}, classicalControl);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 31, HybridStates: {{|10> "
                  "-> 0.71, |11> -> 0.71}: integerValue0 = 0, p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyClassicalControlledGateThatsTrue) {
  std::vector classicalControl = {i0};
  ut.propagateIntAlloc(i0, 1);
  ut.propagateIntAlloc(i1, 0);
  ut.propagateGate(hOp, q1, q5);
  ut.propagateGate(xOp, q3, q6);
  ut.propagateGate(xOp, q6, q7, {}, classicalControl);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 31, HybridStates: {{|00> "
                  "-> 0.71, |01> -> 0.71}: integerValue0 = 1, p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyNegClassicalControlledGateThatsTrue) {
  std::vector classicalControl = {i0};
  ut.propagateIntAlloc(i0, 1);
  ut.propagateIntAlloc(i1, 0);
  ut.propagateGate(hOp, q1, q5);
  ut.propagateGate(xOp, q3, q6);
  ut.propagateGate(xOp, q6, q7, {}, {}, classicalControl);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 31, HybridStates: {{|10> "
                  "-> 0.71, |11> -> 0.71}: integerValue0 = 0, p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyNegClassicalControlledGateThatsFalse) {
  std::vector classicalControl = {i1};
  ut.propagateIntAlloc(i0, 1);
  ut.propagateIntAlloc(i1, 0);
  ut.propagateGate(hOp, q1, q5);
  ut.propagateGate(xOp, q3, q6);
  ut.propagateGate(xOp, q6, q7, {}, classicalControl);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 31, HybridStates: {{|00> "
                  "-> 0.71, |01> -> 0.71}: integerValue0 = 1, p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyPosNegClassicalControlledGateThatsFalse) {
  std::vector classicalControlZero = {i0};
  std::vector classicalControlOne = {i1};
  ut.propagateIntAlloc(i0, 0);
  ut.propagateIntAlloc(i1, 0);
  ut.propagateGate(hOp, q1, q5);
  ut.propagateGate(xOp, q3, q6);
  ut.propagateGate(xOp, q6, q7, {}, classicalControlOne, classicalControlZero);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 31, HybridStates: {{|10> "
                  "-> 0.71, |11> -> 0.71}: integerValue0 = 0, p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyPosNegClassicalControlledGateThatsTrue) {
  std::vector classicalControlTrue = {i0};
  std::vector classicalControlFalse = {i1};
  ut.propagateIntAlloc(i0, 1);
  ut.propagateIntAlloc(i1, 0);
  ut.propagateGate(hOp, q1, q5);
  ut.propagateGate(xOp, q3, q6);
  ut.propagateGate(xOp, q6, q7, classicalControlTrue, classicalControlFalse);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 31, HybridStates: {{|00> "
                  "-> 0.71, |01> -> 0.71}: integerValue0 = 1, p = 1.00;}"));
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
                                 "-> 1.00}: integerValue0 = 1, p = 1.00;}"));
}

TEST_F(UnionTableTest, doMeasurementWithTwoResults) {
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateIntAlloc(i0, 10);
  ut.propagateMeasurement(v5, v6, i0);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 10, HybridStates: {{|00> "
                                 "-> 1.00}: integerValue0 = 0, p = 0.50; {|11> "
                                 "-> 1.00}: integerValue0 = 1, p = 0.50;}"));
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
                  "-> 0.71, |11> -> 0.71}: integerValue0 = 0, p = 1.00;}"));
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
                  "-> 0.71, |11> -> 0.71}: integerValue0 = 10, p = 1.00;}"));
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
  std::vector swapTargets = {v6, v7};
  std::vector swapDestinations = {v8, v9};
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateGate(xOp, q5, q6);
  ut.propagateGate(xOp, q2, q7);
  ut.propagateGate(swapOp, swapTargets, swapDestinations);

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
                                 "|10> -> 0.71}: p = 1.00;}"));
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
  ut.propagateGate(hOp, q1, q3);
  ut.propagateMeasurement(v3, v4, i0);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 0, Bits: 0, HybridStates: {TOP}"));
}

TEST_F(UnionTableWithoutSetupAllocationsTest, doMeasurementsOnTop) {
  auto ut = UnionTable(2, 2);
  ut.propagateQubitAlloc(v0);
  ut.propagateQubitAlloc(v1);
  ut.propagateGate(hOp, q0, q2);
  ut.propagateGate(xOp, q1, q3, q2);
  ut.propagateGate(hOp, q3, q4); // State enters TOP
  ut.propagateMeasurement(v4, v5, i0);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 10, Bits: 0, HybridStates: {TOP}"));
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
              testing::HasSubstr("Qubits: 10, Bits: 0, HybridStates: {TOP}"));
}

TEST_F(UnionTableWithoutSetupAllocationsTest, unifyTooLargeHybridStates) {
  auto ut = UnionTable(4, 2);
  ut.propagateQubitAlloc(v0);
  ut.propagateQubitAlloc(v1);
  ut.propagateQubitAlloc(v2);
  ut.propagateGate(hOp, q0, q3);
  ut.propagateMeasurement(v3, v4, i0);
  ut.propagateGate(xOp, q1, q5, q4);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 10, Bits: 0, HybridStates: {TOP}"));
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
                            {q[0].getType()}, {q[0], q[1], q[2], q[3]});

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
  ut.propagateGate(xOp, q5, q8, {}, {}, ctrl);
  ut.propagateGate(hOp, q6, q9, {}, {}, ctrl);

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
  ut.propagateGate(xOp, q5, q7, {}, {}, ctrl);
  ut.propagateMeasurement(v7, v8, i1);

  EXPECT_FALSE(ut.isClassicalValueAlwaysTrue(v6));
  EXPECT_TRUE(ut.isClassicalValueAlwaysFalse(v8));
}

TEST_F(UnionTablePropertiesTest, bitAlwaysZeroIsFalseOneIsTrue) {
  std::vector ctrl = {i0};
  ut.propagateIntAlloc(i1, 0);
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateMeasurement(v4, v6, i0);
  ut.propagateGate(xOp, q5, q7, {}, {}, {}, ctrl);
  ut.propagateMeasurement(v7, v8, i1);

  EXPECT_TRUE(ut.isClassicalValueAlwaysTrue(v8));
  EXPECT_FALSE(ut.isClassicalValueAlwaysFalse(v6));
}

TEST_F(UnionTablePropertiesTest, testAllTop) {
  ut.propagateGate(hOp, q0, q4);
  ut.propagateGate(xOp, q1, q5, q4);
  ut.propagateGate(xOp, q2, q6, q5);
  ut.propagateMeasurement(v6, v7, i0);
  ut.propagateGate(hOp, q4, q8);
  EXPECT_FALSE(ut.areStatesAllTop());
  ut.propagateGate(hOp, q5, q9);
  EXPECT_FALSE(ut.areStatesAllTop());
  ut.propagateMeasurement(v8, v10, i0);
  EXPECT_TRUE(ut.areStatesAllTop());
}

TEST_F(UnionTablePropertiesTest, testOneGlobalPhase) {
  std::vector qCtrl = {v3, v4};
  ut.propagateGate(xOp, q0, q3);
  ut.propagateGate(xOp, q1, q4);
  ut.propagateGate(xOp, q2, q5);
  auto globalPhase = ut.globalPhaseThatIsAdded(zOp, q5, qCtrl);
  EXPECT_TRUE(globalPhase.has_value());
  EXPECT_EQ(-1, globalPhase.value());
}

TEST_F(UnionTablePropertiesTest, testMinusOneGlobalPhase) {
  std::vector qCtrl = {v4, v5};
  ut.propagateGate(xOp, q0, q4);
  ut.propagateGate(xOp, q1, q5);
  auto emptyGlobalPhase = ut.globalPhaseThatIsAdded(zOp, q5, q4);
  auto globalPhase = ut.globalPhaseThatIsAdded(zOp, q2, qCtrl);
  EXPECT_FALSE(emptyGlobalPhase.has_value());
  EXPECT_TRUE(globalPhase.has_value());
  EXPECT_EQ(1, globalPhase.value());
}

class SmallUnionTableTest : public ::testing::Test {
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

  EXPECT_THAT(
      ut.toString(),
      testing::HasSubstr("Qubits: 32, HybridStates: {{TOP}: p = 1.00;}"));
}

TEST_F(SmallUnionTableTest, applyGatesOnPartiallyTopQState) {
  ut.propagateGate(hOp, q2, q4);
  ut.propagateGate(hOp, q3, q5);
  ut.propagateGate(xOp, q4, q6, q5); // Qubit 2 and 3 enter TOP
  ut.propagateGate(xOp, q1, q7);

  EXPECT_THAT(
      ut.toString(),
      testing::HasSubstr("Qubits: 321, HybridStates: {{TOP}: p = 1.00;}"));
}
