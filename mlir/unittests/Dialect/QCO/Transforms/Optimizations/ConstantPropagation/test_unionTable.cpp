/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ConstantPropagation/UnionTable.hpp"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <gtest/gtest.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

#include <array>
#include <cstddef>
#include <numbers>
#include <string>

namespace {

using namespace mlir;
using namespace mlir::qco;

std::string printed(const UnionTable& ut) {
  std::string s;
  llvm::raw_string_ostream os(s);
  ut.print(os);
  return s;
}

class UnionTableTest : public testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder;

  std::array<Value, 8> q{};
  HOp hOp;
  XOp xOp;
  DCXOp dcxOp;

  UnionTableTest() : builder(&context) {}

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    builder.initialize();
    auto reg = builder.allocQubitRegister(8);
    for (size_t i = 0; i < q.size(); ++i) {
      q[i] = reg[i];
    }
    const auto qt = q[0].getType();
    hOp = HOp::create(builder, builder.getLoc(), qt, q[0]);
    xOp = XOp::create(builder, builder.getLoc(), qt, q[0]);
    dcxOp = DCXOp::create(builder, builder.getLoc(), qt, qt, q[0], q[1]);
  }

  static UnionTable make(const size_t maxAmplitudes = 16,
                         const size_t maxHybridStates = 8) {
    return {maxAmplitudes, maxHybridStates};
  }
};

//===----------------------------------------------------------------------===//
// Seeding
//===----------------------------------------------------------------------===//

TEST_F(UnionTableTest, seedQubitStartsInZero) {
  auto ut = make();
  ut.seedQubit(q[0]);
  EXPECT_TRUE(ut.isTracked(q[0]));
  EXPECT_TRUE(ut.isQubitAlwaysZero(q[0]));
  EXPECT_FALSE(ut.isQubitAlwaysOne(q[0]));
  EXPECT_FALSE(ut.areStatesAllTop());
}

TEST_F(UnionTableTest, seedQubitCantBeCalledTwice) {
  auto ut = make();
  ut.seedQubit(q[0]);
  ASSERT_TRUE(ut.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ut.seedQubit(q[0]); // must not reset the qubit back to |0>
  EXPECT_TRUE(ut.isQubitAlwaysOne(q[0]));
}

TEST_F(UnionTableTest, seedClassicalRecordsConstant) {
  auto ut = make();
  const Value c = builder.boolConstant(true);
  ut.seedClassical(c, builder.getBoolAttr(true));
  EXPECT_TRUE(ut.isTracked(c));
  EXPECT_TRUE(ut.isClassicalAlwaysTrue(c));
  EXPECT_FALSE(ut.isClassicalAlwaysFalse(c));
}

TEST_F(UnionTableTest, untrackedValueQueriesAreFalse) {
  const auto ut = make();
  EXPECT_FALSE(ut.isTracked(q[0]));
  EXPECT_FALSE(ut.isQubitAlwaysZero(q[0]));
  EXPECT_FALSE(ut.isClassicalAlwaysTrue(q[0]));
}

//===----------------------------------------------------------------------===//
// Factorisation
//===----------------------------------------------------------------------===//

TEST_F(UnionTableTest, independentQubitsStayFactored) {
  auto ut = make();
  ut.seedQubit(q[0]);
  ut.seedQubit(q[1]);
  ASSERT_TRUE(ut.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  EXPECT_TRUE(ut.isQubitAlwaysOne(q[0]));
  EXPECT_TRUE(ut.isQubitAlwaysZero(q[1]));
  // Two independent factors print on two lines (a coalesced pair would be one).
  EXPECT_NE(printed(ut).find('\n'), std::string::npos);
}

TEST_F(UnionTableTest, twoQubitGateMergeTargets) {
  auto ut = make();
  ut.seedQubit(q[0]);
  ut.seedQubit(q[1]);
  ASSERT_TRUE(ut.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(ut.applyMatrix2Q(q[0], q[1], q[0], q[1], dcxOp.getUnitaryMatrix())
                  .succeeded());
  EXPECT_TRUE(ut.isQubitAlwaysZero(q[0]));
  EXPECT_TRUE(ut.isQubitAlwaysOne(q[1]));
}

TEST_F(UnionTableTest, controlledGateFiresAcrossSlots) {
  auto ut = make();
  ut.seedQubit(q[0]);
  ut.seedQubit(q[1]);
  ASSERT_TRUE(ut.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(
      ut.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix(), {q[0]}, {q[0]})
          .succeeded());
  EXPECT_TRUE(ut.isQubitAlwaysOne(q[1]));
}

TEST_F(UnionTableTest, controlledGateDoesNotFireWhenControlIsZero) {
  auto ut = make();
  ut.seedQubit(q[0]);
  ut.seedQubit(q[1]);
  ASSERT_TRUE(
      ut.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix(), {q[0]}, {q[0]})
          .succeeded());
  EXPECT_TRUE(ut.isQubitAlwaysZero(q[1]));
}

TEST_F(UnionTableTest, applyToUnseededQubitFails) {
  auto ut = make();
  ut.seedQubit(q[0]);
  EXPECT_TRUE(ut.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix()).failed());
}

//===----------------------------------------------------------------------===//
// Classical controls
//===----------------------------------------------------------------------===//

TEST_F(UnionTableTest, classicalControlSkipsGate) {
  auto ut = make();
  ut.seedQubit(q[0]);
  const Value c = builder.boolConstant(false);
  ut.seedClassical(c, builder.getBoolAttr(false));
  ASSERT_TRUE(ut.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix(), {}, {}, {c})
                  .succeeded());
  EXPECT_TRUE(ut.isQubitAlwaysZero(q[0]));
}

TEST_F(UnionTableTest, unresolvedClassicalControlFails) {
  auto ut = make();
  ut.seedQubit(q[0]);
  const Value c =
      builder.boolConstant(false);
  EXPECT_TRUE(ut.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix(), {}, {}, {c})
                  .failed());
}

//===----------------------------------------------------------------------===//
// Measurement / reset
//===----------------------------------------------------------------------===//

TEST_F(UnionTableTest, measureDeterministicRecordsBit) {
  auto ut = make();
  ut.seedQubit(q[0]);
  const Value result = builder.boolConstant(false);
  ASSERT_TRUE(ut.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(ut.measureQubit(q[0], q[0], result).succeeded());
  EXPECT_TRUE(ut.isClassicalAlwaysTrue(result));
}

TEST_F(UnionTableTest, measureSuperpositionTopsTheState) {
  auto ut = make();
  ut.seedQubit(q[0]);
  const Value result = builder.boolConstant(false);
  ASSERT_TRUE(ut.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(ut.measureQubit(q[0], q[0], result).succeeded());
  EXPECT_TRUE(ut.areStatesAllTop());
  EXPECT_FALSE(ut.isClassicalAlwaysTrue(result));
}

TEST_F(UnionTableTest, resetForcesZero) {
  auto ut = make();
  ut.seedQubit(q[0]);
  ASSERT_TRUE(ut.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(ut.resetQubit(q[0], q[0]).succeeded());
  EXPECT_TRUE(ut.isQubitAlwaysZero(q[0]));
}

//===----------------------------------------------------------------------===//
// Global phase
//===----------------------------------------------------------------------===//

TEST_F(UnionTableTest, globalPhaseIsRecordedOnce) {
  auto ut = make();
  ut.seedQubit(q[0]);
  ASSERT_TRUE(ut.addGlobalPhase(std::numbers::pi).succeeded());
  EXPECT_NE(printed(ut).find("phase="), std::string::npos);
}

//===----------------------------------------------------------------------===//
// Control analysis
//===----------------------------------------------------------------------===//

TEST_F(UnionTableTest, controlsSatisfiableWhenBothCanBeOne) {
  auto ut = make();
  ut.seedQubit(q[0]);
  ut.seedQubit(q[1]);
  ASSERT_TRUE(ut.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(ut.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix()).succeeded());
  EXPECT_TRUE(ut.areControlsSatisfiable({q[0], q[1]}));
}

TEST_F(UnionTableTest, controlsUnsatisfiableWhenAQubitIsAlwaysZero) {
  auto ut = make();
  ut.seedQubit(q[0]);
  ut.seedQubit(q[1]);
  ASSERT_TRUE(ut.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  EXPECT_FALSE(ut.areControlsSatisfiable({q[0], q[1]}));
}

TEST_F(UnionTableTest, negativeClassicalControlSatisfiedByFalseConstant) {
  auto ut = make();
  const Value c = builder.boolConstant(false);
  ut.seedClassical(c, builder.getBoolAttr(false));
  EXPECT_FALSE(ut.areControlsSatisfiable({}, {c}));
  EXPECT_TRUE(ut.areControlsSatisfiable({}, {}, {c}));
}

TEST_F(UnionTableTest, superfluousControlsListsAlwaysOneQubit) {
  auto ut = make();
  ut.seedQubit(q[0]);
  ASSERT_TRUE(ut.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  const auto result = ut.getSuperfluousControls({q[0]});
  EXPECT_FALSE(result.completelySuperfluous);
  EXPECT_TRUE(result.superfluousQubits.contains(q[0]));
}

TEST_F(UnionTableTest, superfluousControlsFlagsDeadGate) {
  auto ut = make();
  ut.seedQubit(q[0]);
  const auto result = ut.getSuperfluousControls({q[0]});
  EXPECT_TRUE(result.completelySuperfluous);
}

//===----------------------------------------------------------------------===//
// markQubitsTop / forwarding
//===----------------------------------------------------------------------===//

TEST_F(UnionTableTest, markQubitsTopClearsQuantumInfo) {
  auto ut = make();
  ut.seedQubit(q[0]);
  ut.markQubitsTop({q[0]});
  EXPECT_TRUE(ut.areStatesAllTop());
  EXPECT_FALSE(ut.isQubitAlwaysZero(q[0]));
}

TEST_F(UnionTableTest, forwardValueRenamesQubit) {
  auto ut = make();
  ut.seedQubit(q[0]);
  ASSERT_TRUE(ut.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ut.forwardValue(q[0], q[1]);
  EXPECT_FALSE(ut.isTracked(q[0]));
  EXPECT_TRUE(ut.isQubitAlwaysOne(q[1]));
}

//===----------------------------------------------------------------------===//
// join
//===----------------------------------------------------------------------===//

TEST_F(UnionTableTest, joinOfAgreeingBranchesKeepsTheFact) {
  auto a = make();
  a.seedQubit(q[0]);
  ASSERT_TRUE(a.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  auto b = make();
  b.seedQubit(q[0]);
  ASSERT_TRUE(b.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());

  a.join(b);
  EXPECT_TRUE(a.isQubitAlwaysOne(q[0]));
  EXPECT_FALSE(a.isAllTop());
}

TEST_F(UnionTableTest, joinOfDisagreeingBranchesIsAProbabilisticSplit) {
  auto a = make();
  a.seedQubit(q[0]);
  auto b = make();
  b.seedQubit(q[0]);
  ASSERT_TRUE(b.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());

  a.join(b);
  EXPECT_FALSE(a.isQubitAlwaysZero(q[0]));
  EXPECT_FALSE(a.isQubitAlwaysOne(q[0]));
  EXPECT_FALSE(a.isAllTop());
  EXPECT_NE(printed(a).find("p=0.5000"), std::string::npos);
}

TEST_F(UnionTableTest, joinOfDifferentEntanglementStructureTops) {
  auto a = make();
  a.seedQubit(q[0]);
  a.seedQubit(q[1]);
  ASSERT_TRUE(a.applyMatrix2Q(q[0], q[1], q[0], q[1], dcxOp.getUnitaryMatrix())
                  .succeeded());
  auto b = make();
  b.seedQubit(q[0]);
  b.seedQubit(q[1]);

  a.join(b);
  EXPECT_TRUE(a.isAllTop());
}

TEST_F(UnionTableTest, joinKeepsClassicalFactOnlyWhenShared) {
  const Value c = builder.boolConstant(true);

  auto a = make();
  a.seedClassical(c, builder.getBoolAttr(true));
  auto agree = make();
  agree.seedClassical(c, builder.getBoolAttr(true));
  a.join(agree);
  EXPECT_TRUE(a.isClassicalAlwaysTrue(c));

  auto d = make();
  d.seedClassical(c, builder.getBoolAttr(true));
  auto disagree = make();
  disagree.seedClassical(c, builder.getBoolAttr(false));
  d.join(disagree);
  EXPECT_FALSE(d.isClassicalAlwaysTrue(c));
  EXPECT_FALSE(d.isClassicalAlwaysFalse(c));
}

TEST_F(UnionTableTest, joinOverflowingAFactorTopsOnlyThatFactor) {
  auto a = make(16, 2);
  a.seedQubit(q[0]);
  a.seedQubit(q[1]);
  auto b = make(16, 2);
  b.seedQubit(q[0]);
  b.seedQubit(q[1]);
  ASSERT_TRUE(b.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  a.join(b);
  ASSERT_FALSE(a.isAllTop());

  auto c = make(16, 2);
  c.seedQubit(q[0]);
  c.seedQubit(q[1]);
  ASSERT_TRUE(c.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  a.join(c);

  EXPECT_FALSE(a.isAllTop());
  EXPECT_FALSE(a.isQubitAlwaysZero(q[0])); // {q0} factor collapsed to top
  EXPECT_TRUE(a.isQubitAlwaysZero(q[1]));  // {q1} factor reconciled normally
}

//===----------------------------------------------------------------------===//
// Equality
//===----------------------------------------------------------------------===//

TEST_F(UnionTableTest, equalityIsOrderIndependent) {
  auto a = make();
  a.seedQubit(q[0]);
  a.seedQubit(q[1]);
  auto b = make();
  b.seedQubit(q[1]);
  b.seedQubit(q[0]);
  EXPECT_TRUE(a == b);
}

TEST_F(UnionTableTest, equalitySeesAppliedGates) {
  auto a = make();
  a.seedQubit(q[0]);
  auto b = make();
  b.seedQubit(q[0]);
  ASSERT_TRUE(b.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  EXPECT_FALSE(a == b);
}

TEST_F(UnionTableTest, markAllTopIsAbsorbing) {
  auto ut = make();
  ut.seedQubit(q[0]);
  ut.markAllTop();
  EXPECT_TRUE(ut.isAllTop());
  EXPECT_TRUE(ut.areStatesAllTop());
  EXPECT_EQ(printed(ut), "<all top>");
  ut.seedQubit(q[1]);
  EXPECT_FALSE(ut.isTracked(q[1]));
}

} // namespace
