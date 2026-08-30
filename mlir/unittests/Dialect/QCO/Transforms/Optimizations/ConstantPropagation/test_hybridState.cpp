/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ConstantPropagation/HybridState.hpp"
#include "ConstantPropagation/QuantumState.hpp"
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
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>

using namespace mlir;
using namespace mlir::qco;

static std::string printed(const HybridState& hs) {
  std::string s;
  llvm::raw_string_ostream os(s);
  hs.print(os);
  return s;
}

namespace {

constexpr size_t BUDGET = 16;

class HybridStateTest : public testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder;

  std::array<Value, 4> q{};
  Value cA;
  Value cB;
  HOp hOp;
  XOp xOp;
  DCXOp dcxOp;

  HybridStateTest() : builder(&context) {}

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    builder.initialize();
    auto reg = builder.allocQubitRegister(4);
    for (size_t i = 0; i < q.size(); ++i) {
      q[i] = reg[i];
    }
    cA = builder.boolConstant(false);
    cB = builder.boolConstant(true);
    const auto qt = q[0].getType();
    hOp = HOp::create(builder, builder.getLoc(), qt, q[0]);
    xOp = XOp::create(builder, builder.getLoc(), qt, q[0]);
    dcxOp = DCXOp::create(builder, builder.getLoc(), qt, qt, q[0], q[1]);
  }

  static HybridState make(ArrayRef<Value> qubits, double probability = 1.0) {
    return HybridState(QuantumState(qubits, BUDGET), BUDGET, probability);
  }
};

//===----------------------------------------------------------------------===//
// Construction / classical values
//===----------------------------------------------------------------------===//

TEST_F(HybridStateTest, holdsItsQubits) {
  const auto hs = make({q[0], q[1]});
  EXPECT_TRUE(hs.hasQubit(q[0]));
  EXPECT_FALSE(hs.hasQubit(q[2]));
  EXPECT_EQ(hs.getQubits().size(), 2U);
  EXPECT_TRUE(hs.isQubitAlwaysZero(q[0]));
  EXPECT_FALSE(hs.isTop());
}

TEST_F(HybridStateTest, setAndGetClassical) {
  auto hs = make({});
  EXPECT_FALSE(hs.getClassical(cA).has_value());

  hs.setClassical(cA, builder.getBoolAttr(true));
  ASSERT_TRUE(hs.getClassical(cA).has_value());
  EXPECT_TRUE(hs.isClassicalTrue(cA));
  EXPECT_FALSE(hs.isClassicalFalse(cA));

  hs.setClassical(cA, builder.getBoolAttr(false));
  EXPECT_FALSE(hs.isClassicalTrue(cA));
  EXPECT_TRUE(hs.isClassicalFalse(cA));
}

//===----------------------------------------------------------------------===//
// Gate application
//===----------------------------------------------------------------------===//

TEST_F(HybridStateTest, applyMatrix1Q) {
  auto hs = make({q[0]});
  ASSERT_TRUE(hs.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  EXPECT_FALSE(hs.isQubitAlwaysZero(q[0]));
}

TEST_F(HybridStateTest, applyToQubitNotInStateFails) {
  auto hs = make({q[0]});
  EXPECT_TRUE(hs.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix()).failed());
}

TEST_F(HybridStateTest, applyMatrix2Q) {
  auto hs = make({q[0], q[1]});
  ASSERT_TRUE(hs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(hs.applyMatrix2Q(q[0], q[1], q[0], q[1], dcxOp.getUnitaryMatrix())
                  .succeeded());
  EXPECT_TRUE(hs.isQubitAlwaysZero(q[0]));
  EXPECT_TRUE(hs.isQubitAlwaysOne(q[1]));
}

TEST_F(HybridStateTest, quantumControlledGate) {
  auto hs = make({q[0], q[1]});
  ASSERT_TRUE(hs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(
      hs.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix(), {q[0]}, {q[0]})
          .succeeded());
  EXPECT_TRUE(hs.isQubitAlwaysOne(q[1]));
}

TEST_F(HybridStateTest, controlRenameUpdatesState) {
  auto hs = make({q[0], q[1]});
  ASSERT_TRUE(
      hs.applyMatrix1Q(q[1], q[3], xOp.getUnitaryMatrix(), {q[0]}, {q[2]})
          .succeeded());
  EXPECT_FALSE(hs.hasQubit(q[0]));
  EXPECT_FALSE(hs.hasQubit(q[1]));
  EXPECT_TRUE(hs.hasQubit(q[2]));
  EXPECT_TRUE(hs.hasQubit(q[3]));
}

TEST_F(HybridStateTest, controlInOutLengthMismatchFails) {
  auto hs = make({q[0], q[1]});
  EXPECT_TRUE(
      hs.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix(), {q[0]}, {q[2], q[3]})
          .failed());
}

TEST_F(HybridStateTest, nonEmptyControlInEmptyControlOutFails) {
  auto hs = make({q[0], q[1]});
  EXPECT_TRUE(
      hs.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix(), {q[0]}, {}).failed());
}

//===----------------------------------------------------------------------===//
// Classical controls
//===----------------------------------------------------------------------===//

TEST_F(HybridStateTest, positiveClassicalControlHoldsAppliesGate) {
  auto hs = make({q[0]});
  hs.setClassical(cA, builder.getBoolAttr(true));
  ASSERT_TRUE(hs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix(), {}, {}, {cA})
                  .succeeded());
  EXPECT_TRUE(hs.isQubitAlwaysOne(q[0]));
}

TEST_F(HybridStateTest, positiveClassicalControlFailsSkipsGateButRenames) {
  auto hs = make({q[0]});
  hs.setClassical(cA, builder.getBoolAttr(false));
  ASSERT_TRUE(hs.applyMatrix1Q(q[0], q[1], xOp.getUnitaryMatrix(), {}, {}, {cA})
                  .succeeded());
  EXPECT_FALSE(hs.hasQubit(q[0]));
  EXPECT_TRUE(hs.hasQubit(q[1]));
  EXPECT_TRUE(hs.isQubitAlwaysZero(q[1]));
}

TEST_F(HybridStateTest, negativeClassicalControlHoldsAppliesGate) {
  auto hs = make({q[0]});
  hs.setClassical(cA, builder.getBoolAttr(false));
  ASSERT_TRUE(
      hs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix(), {}, {}, {}, {cA})
          .succeeded());
  EXPECT_TRUE(hs.isQubitAlwaysOne(q[0]));
}

TEST_F(HybridStateTest, negativeClassicalControlFailsSkipsGate) {
  auto hs = make({q[0]});
  hs.setClassical(cA, builder.getBoolAttr(true));
  ASSERT_TRUE(
      hs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix(), {}, {}, {}, {cA})
          .succeeded());
  EXPECT_TRUE(hs.isQubitAlwaysZero(q[0]));
}

TEST_F(HybridStateTest, unresolvedClassicalControlFails) {
  auto hs = make({q[0]});
  EXPECT_TRUE(hs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix(), {}, {}, {cA})
                  .failed());
}

TEST_F(HybridStateTest, floatClassicalControlIsSupported) {
  auto hs = make({q[0]});
  Value fc = builder.floatConstant(2.5);

  hs.setClassical(fc, builder.getF64FloatAttr(2.5));
  EXPECT_TRUE(hs.isClassicalTrue(fc));
  ASSERT_TRUE(hs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix(), {}, {}, {fc})
                  .succeeded());
  EXPECT_TRUE(hs.isQubitAlwaysOne(q[0]));

  hs.setClassical(fc, builder.getF64FloatAttr(0.0));
  EXPECT_TRUE(hs.isClassicalFalse(fc));
}

//===----------------------------------------------------------------------===//
// Global phase
//===----------------------------------------------------------------------===//

TEST_F(HybridStateTest, uncontrolledGlobalPhaseAccumulates) {
  auto hs = make({q[0]});
  Value theta = builder.floatConstant(std::acos(-1.0));
  hs.setClassical(theta, builder.getF64FloatAttr(std::acos(-1.0)));
  ASSERT_TRUE(hs.addGlobalPhase(theta).succeeded());
  EXPECT_LT(std::abs(hs.getGlobalPhase() - Complex{-1.0, 0.0}), 1e-9);
}

TEST_F(HybridStateTest, quantumControlledPhaseIsNotGlobal) {
  auto hs = make({q[0], q[1]});
  Value theta = builder.floatConstant(std::acos(-1.0));
  hs.setClassical(theta, builder.getF64FloatAttr(std::acos(-1.0)));
  ASSERT_TRUE(hs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(hs.addGlobalPhase(theta, {q[0]}, {q[0]}).succeeded());
  EXPECT_LT(std::abs(hs.getGlobalPhase() - Complex{1.0, 0.0}), 1e-9);
}

TEST_F(HybridStateTest, globalPhaseSkippedByClassicalControl) {
  auto hs = make({q[0]});
  Value theta = builder.floatConstant(std::acos(-1.0));
  hs.setClassical(theta, builder.getF64FloatAttr(std::acos(-1.0)));
  hs.setClassical(cA, builder.getBoolAttr(false));
  ASSERT_TRUE(hs.addGlobalPhase(theta, {}, {}, {cA}).succeeded());
  EXPECT_LT(std::abs(hs.getGlobalPhase() - Complex{1.0, 0.0}), 1e-9);
}

TEST_F(HybridStateTest, globalPhaseFailsWhenThetaUnresolved) {
  auto hs = make({q[0]});
  Value theta = builder.floatConstant(std::acos(-1.0)); // never seeded
  EXPECT_TRUE(hs.addGlobalPhase(theta).failed());
}

TEST_F(HybridStateTest, propagateClassicalFoldsConstants) {
  auto hs = make({});
  Value lhs = builder.intConstant(3);
  Value rhs = builder.intConstant(4);
  hs.setClassical(lhs, builder.getIntegerAttr(lhs.getType(), 3));
  hs.setClassical(rhs, builder.getIntegerAttr(rhs.getType(), 4));
  auto add = arith::AddIOp::create(builder, builder.getLoc(), lhs, rhs);
  hs.propagateClassical(add.getOperation());
  const auto folded = hs.getClassical(add.getResult());
  ASSERT_TRUE(folded.has_value());
  const auto intAttr = dyn_cast<IntegerAttr>(*folded);
  ASSERT_TRUE(intAttr);
  EXPECT_EQ(intAttr.getInt(), 7);
}

//===----------------------------------------------------------------------===//
// tensor
//===----------------------------------------------------------------------===//

TEST_F(HybridStateTest, tensorCombinesDisjointSubsystems) {
  auto a = make({q[0]});
  ASSERT_TRUE(a.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  a.setClassical(cA, builder.getBoolAttr(true));
  auto b = make({q[1]});
  b.setClassical(cB, builder.getBoolAttr(false));

  const auto ab = a.tensor(b);
  EXPECT_EQ(ab.getQubits().size(), 2U);
  EXPECT_TRUE(ab.hasQubit(q[0]));
  EXPECT_TRUE(ab.hasQubit(q[1]));
  EXPECT_TRUE(ab.isClassicalTrue(cA));
  EXPECT_TRUE(ab.isClassicalFalse(cB));
}

TEST_F(HybridStateTest, tensorMultipliesProbabilities) {
  const auto a = make({q[0]}, 0.5);
  const auto b = make({q[1]}, 0.5);
  EXPECT_DOUBLE_EQ(a.tensor(b).getProbability(), 0.25);
}

//===----------------------------------------------------------------------===//
// Measurement
//===----------------------------------------------------------------------===//

TEST_F(HybridStateTest, measureDeterministicRecordsClassical) {
  auto hs = make({q[0]});
  ASSERT_TRUE(hs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(hs.measureQubit(q[0], q[1], cA).succeeded());
  EXPECT_TRUE(hs.isClassicalTrue(cA));
  EXPECT_TRUE(hs.hasQubit(q[1]));
  EXPECT_FALSE(hs.hasQubit(q[0]));
}

TEST_F(HybridStateTest, measureSuperpositionTopsAndLeavesResultUnknown) {
  auto hs = make({q[0]});
  ASSERT_TRUE(hs.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(hs.measureQubit(q[0], q[1], cA).succeeded());
  EXPECT_FALSE(hs.getClassical(cA).has_value());
  EXPECT_TRUE(hs.isTop());
}

TEST_F(HybridStateTest, measureSkippedByClassicalControl) {
  auto hs = make({q[0]});
  hs.setClassical(cB, builder.getBoolAttr(false));
  ASSERT_TRUE(hs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(hs.measureQubit(q[0], q[1], cA, {cB}).succeeded());
  EXPECT_FALSE(hs.getClassical(cA).has_value());
  EXPECT_TRUE(hs.hasQubit(q[1]));
}

TEST_F(HybridStateTest, measureUnseededFails) {
  auto hs = make({q[0]});
  EXPECT_TRUE(hs.measureQubit(q[1], q[2], cA).failed());
}

//===----------------------------------------------------------------------===//
// Reset
//===----------------------------------------------------------------------===//

TEST_F(HybridStateTest, resetSingletonForcesZero) {
  auto hs = make({q[0]});
  ASSERT_TRUE(hs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(hs.resetQubit(q[0], q[1]).succeeded());
  EXPECT_TRUE(hs.isQubitAlwaysZero(q[1]));
}

TEST_F(HybridStateTest, resetDeterministicOneInLargerState) {
  auto hs = make({q[0], q[1]});
  ASSERT_TRUE(hs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(
      hs.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix(), {q[0]}, {q[0]})
          .succeeded());
  ASSERT_TRUE(hs.resetQubit(q[0], q[2]).succeeded());
  EXPECT_TRUE(hs.isQubitAlwaysZero(q[2]));
  EXPECT_TRUE(hs.isQubitAlwaysOne(q[1]));
  EXPECT_FALSE(hs.isTop());
}

TEST_F(HybridStateTest, resetSuperpositionInLargerStateTops) {
  auto hs = make({q[0], q[1]});
  ASSERT_TRUE(hs.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(
      hs.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix(), {q[0]}, {q[0]})
          .succeeded());
  ASSERT_TRUE(hs.resetQubit(q[0], q[2]).succeeded());
  EXPECT_TRUE(hs.isTop());
}

TEST_F(HybridStateTest, resetUnseededFails) {
  auto hs = make({q[0]});
  EXPECT_TRUE(hs.resetQubit(q[1], q[2]).failed());
}

//===----------------------------------------------------------------------===//
// Control satisfiability
//===----------------------------------------------------------------------===//

TEST_F(HybridStateTest, controlsSatisfiableClassical) {
  auto hs = make({});
  hs.setClassical(cA, builder.getBoolAttr(true));
  hs.setClassical(cB, builder.getBoolAttr(false));
  EXPECT_TRUE(hs.areControlsSatisfiable({}, {cA}, {cB}));
  EXPECT_FALSE(hs.areControlsSatisfiable({}, {cB}, {}));
  EXPECT_FALSE(hs.areControlsSatisfiable({}, {}, {cA}));
}

TEST_F(HybridStateTest, controlsSatisfiableQuantum) {
  auto hs = make({q[0], q[1]});
  ASSERT_TRUE(hs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  EXPECT_TRUE(hs.areControlsSatisfiable({q[0]}, {}, {}));
  EXPECT_FALSE(hs.areControlsSatisfiable({q[1]}, {}, {}));

  ASSERT_TRUE(hs.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix()).succeeded());
  EXPECT_TRUE(hs.areControlsSatisfiable({q[0], q[1]}, {}, {}));
}

TEST_F(HybridStateTest, controlsSatisfiableQubitNotInStateIsFalse) {
  const auto hs = make({q[0]});
  EXPECT_FALSE(hs.areControlsSatisfiable({q[1]}, {}, {}));
}

//===----------------------------------------------------------------------===//
// Equality / print
//===----------------------------------------------------------------------===//

TEST_F(HybridStateTest, equalityConsidersEverything) {
  auto a = make({q[0]}, 0.5);
  auto b = make({q[0]}, 0.5);
  EXPECT_TRUE(a == b);

  EXPECT_FALSE(a == make({q[0]}, 0.25));

  b.setClassical(cA, builder.getBoolAttr(true));
  EXPECT_FALSE(a == b);

  auto c = make({q[0]}, 0.5);
  ASSERT_TRUE(c.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  EXPECT_FALSE(a == c);
}

TEST_F(HybridStateTest, sameConfigurationIgnoresProbability) {
  auto a = make({q[0]}, 0.5);
  const auto b = make({q[0]}, 0.25);
  EXPECT_TRUE(a.sameConfiguration(b));
  EXPECT_FALSE(a == b);

  a.setClassical(cA, builder.getBoolAttr(true));
  EXPECT_FALSE(a.sameConfiguration(b));
}

TEST_F(HybridStateTest, setProbabilityReplacesTheWeight) {
  auto hs = make({q[0]}, 0.5);
  hs.setProbability(0.2);
  EXPECT_DOUBLE_EQ(hs.getProbability(), 0.2);
}

TEST_F(HybridStateTest, markStateTopKeepsClassicalFacts) {
  auto hs = make({q[0]});
  hs.setClassical(cA, builder.getBoolAttr(true));
  hs.markStateTop();
  EXPECT_TRUE(hs.isTop());
  EXPECT_TRUE(hs.isClassicalTrue(cA));
}

TEST_F(HybridStateTest, intersectClassicalKeepsOnlyAgreedFacts) {
  auto a = make({q[0]});
  a.setClassical(cA, builder.getBoolAttr(true));
  a.setClassical(cB, builder.getBoolAttr(true));

  auto b = make({q[0]});
  b.setClassical(cA, builder.getBoolAttr(true));  // agrees
  b.setClassical(cB, builder.getBoolAttr(false)); // disagrees

  a.intersectClassical(b);
  EXPECT_TRUE(a.isClassicalTrue(cA));
  EXPECT_FALSE(a.getClassical(cB).has_value());
}

TEST_F(HybridStateTest, forwardValueRenamesQubitAndClassical) {
  auto hs = make({q[0]});
  hs.setClassical(cA, builder.getBoolAttr(true));
  hs.forwardValue(q[0], q[1]);
  hs.forwardValue(cA, cB);
  EXPECT_FALSE(hs.hasQubit(q[0]));
  EXPECT_TRUE(hs.hasQubit(q[1]));
  EXPECT_FALSE(hs.getClassical(cA).has_value());
  EXPECT_TRUE(hs.isClassicalTrue(cB));
}

TEST_F(HybridStateTest, printIsNonEmpty) {
  auto hs = make({q[0]});
  hs.setClassical(cA, builder.getBoolAttr(false));
  EXPECT_NE(printed(hs).find("p=1.0000"), std::string::npos);
}

TEST_F(HybridStateTest, globalPhaseAcceptsIntegerTheta) {
  auto hs = make({q[0]});
  Value theta = builder.intConstant(3);
  hs.setClassical(theta, builder.getIntegerAttr(theta.getType(), 3));
  ASSERT_TRUE(hs.addGlobalPhase(theta).succeeded());
  EXPECT_LT(std::abs(hs.getGlobalPhase() - std::polar(1.0, 3.0)), 1e-9);
}

TEST_F(HybridStateTest,
       twoQubitGateSkippedByFalseClassicalControlStillForwards) {
  auto hs = make({q[0], q[1]});
  ASSERT_TRUE(hs.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix()).succeeded());

  hs.setClassical(cA, builder.getBoolAttr(false));
  ASSERT_TRUE(hs.applyMatrix2Q(q[0], q[1], q[2], q[3], dcxOp.getUnitaryMatrix(),
                               {}, {}, {cA})
                  .succeeded());
  EXPECT_FALSE(hs.hasQubit(q[0]));
  EXPECT_TRUE(hs.hasQubit(q[2]));
  EXPECT_TRUE(hs.hasQubit(q[3]));
  EXPECT_TRUE(hs.isQubitAlwaysZero(q[2]));
  EXPECT_TRUE(hs.isQubitAlwaysOne(q[3]));
}

TEST_F(HybridStateTest, unresolvedNegativeClassicalControlFails) {
  auto hs = make({q[0]});
  EXPECT_TRUE(
      hs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix(), {}, {}, {}, {cA})
          .failed());
}

} // namespace
