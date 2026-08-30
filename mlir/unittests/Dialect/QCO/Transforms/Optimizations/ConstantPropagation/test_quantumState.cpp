/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ConstantPropagation/QuantumState.hpp"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>

namespace {

using namespace mlir;
using namespace mlir::qco;

/// Renders a QuantumState through its print() method for readable assertions.
std::string printed(const QuantumState& qs) {
  std::string s;
  llvm::raw_string_ostream os(s);
  qs.print(os);
  return s;
}

class QuantumStateTest : public testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder;

  std::array<Value, 4> q{};
  HOp hOp;
  XOp xOp;
  ZOp zOp;
  SWAPOp swapOp;
  DCXOp dcxOp;

  QuantumStateTest() : builder(&context) {}

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
    const auto qt = q[0].getType();
    hOp = HOp::create(builder, builder.getLoc(), qt, q[0]);
    xOp = XOp::create(builder, builder.getLoc(), qt, q[0]);
    zOp = ZOp::create(builder, builder.getLoc(), qt, q[0]);
    swapOp = SWAPOp::create(builder, builder.getLoc(), qt, qt, q[0], q[1]);
    dcxOp = DCXOp::create(builder, builder.getLoc(), qt, qt, q[0], q[1]);
  }
};

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

TEST_F(QuantumStateTest, allZeroState) {
  const auto qs = QuantumState({q[0], q[1], q[2], q[3]}, 4);
  EXPECT_FALSE(qs.isTop());
  EXPECT_EQ(qs.getQubits().size(), 4U);
  EXPECT_EQ(printed(qs), "|0000> -> 1.00");
}

//===----------------------------------------------------------------------===//
// Single-qubit gates
//===----------------------------------------------------------------------===//

TEST_F(QuantumStateTest, applyH) {
  auto qs = QuantumState::singletonZero(q[0], 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  EXPECT_EQ(printed(qs), "|0> -> 0.71, |1> -> 0.71");
}

TEST_F(QuantumStateTest, applyHToThirdQubit) {
  auto qs = QuantumState({q[0], q[1], q[2], q[3]}, 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[2], q[2], hOp.getUnitaryMatrix()).succeeded());
  EXPECT_EQ(printed(qs), "|0000> -> 0.71, |0100> -> 0.71");
}

TEST_F(QuantumStateTest, applyHTwiceIsIdentity) {
  auto qs = QuantumState({q[0], q[1], q[2], q[3]}, 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[2], q[2], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(qs.applyMatrix1Q(q[2], q[2], hOp.getUnitaryMatrix()).succeeded());
  EXPECT_EQ(printed(qs), "|0000> -> 1.00");
}

TEST_F(QuantumStateTest, applyHThenZ) {
  auto qs = QuantumState({q[0], q[1], q[2], q[3]}, 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[2], q[2], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(qs.applyMatrix1Q(q[2], q[2], zOp.getUnitaryMatrix()).succeeded());
  EXPECT_EQ(printed(qs), "|0000> -> 0.71, |0100> -> -0.71");
}

TEST_F(QuantumStateTest, applyHZHIsX) {
  auto qs = QuantumState({q[0], q[1], q[2], q[3]}, 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[2], q[2], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(qs.applyMatrix1Q(q[2], q[2], zOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(qs.applyMatrix1Q(q[2], q[2], hOp.getUnitaryMatrix()).succeeded());
  EXPECT_EQ(printed(qs), "|0100> -> 1.00");
}

TEST_F(QuantumStateTest, applyGatesToTwoIndependentQubits) {
  auto qs = QuantumState({q[0], q[1], q[2], q[3]}, 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[2], q[2], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  EXPECT_EQ(printed(qs), "|0001> -> 0.71, |0101> -> 0.71");
}

TEST_F(QuantumStateTest, forwardQubitRenamesInPlace) {
  auto qs = QuantumState::singletonZero(q[0], 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[1], xOp.getUnitaryMatrix()).succeeded());
  EXPECT_FALSE(qs.contains(q[0]));
  EXPECT_TRUE(qs.contains(q[1]));
  EXPECT_TRUE(qs.isAlwaysOne(q[1]));
}

//===----------------------------------------------------------------------===//
// Two-qubit gates
//===----------------------------------------------------------------------===//

TEST_F(QuantumStateTest, applySwap) {
  auto qs = QuantumState({q[0], q[1], q[2], q[3]}, 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[1], q[1], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(
      qs.applyMatrix2Q(q[1], q[3], q[1], q[3], swapOp.getUnitaryMatrix())
          .succeeded());
  EXPECT_EQ(printed(qs), "|0000> -> 0.71, |1000> -> 0.71");
}

TEST_F(QuantumStateTest, applyDcxActsAsCxCx) {
  auto qs = QuantumState({q[0], q[1]}, 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(qs.applyMatrix2Q(q[0], q[1], q[0], q[1], dcxOp.getUnitaryMatrix())
                  .succeeded());
  EXPECT_TRUE(qs.isAlwaysZero(q[0]));
  EXPECT_TRUE(qs.isAlwaysOne(q[1]));
  EXPECT_EQ(printed(qs), "|10> -> 1.00");
}

//===----------------------------------------------------------------------===//
// Precondition failures
//===----------------------------------------------------------------------===//

TEST_F(QuantumStateTest, applyToQubitNotInGroupFails) {
  auto qs = QuantumState::singletonZero(q[0], 4);
  EXPECT_TRUE(qs.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix()).failed());
}

TEST_F(QuantumStateTest, applyTwoQubitGateToSameBitFails) {
  auto qs = QuantumState({q[0], q[1]}, 4);
  EXPECT_TRUE(
      qs.applyMatrix2Q(q[0], q[0], q[0], q[0], swapOp.getUnitaryMatrix())
          .failed());
}

TEST_F(QuantumStateTest, applyWithControlNotInGroupFails) {
  auto qs = QuantumState({q[0], q[1]}, 4);
  EXPECT_TRUE(
      qs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix(), {q[2]}).failed());
}

TEST_F(QuantumStateTest, applyToQubitNotInGroupFailsEvenWhenTop) {
  auto qs = QuantumState({q[0], q[1], q[2], q[3]}, 1);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(qs.isTop());
  EXPECT_TRUE(qs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  Value stranger = builder.allocQubit();
  EXPECT_TRUE(
      qs.applyMatrix1Q(stranger, stranger, xOp.getUnitaryMatrix()).failed());
}

//===----------------------------------------------------------------------===//
// Controls
//===----------------------------------------------------------------------===//

TEST_F(QuantumStateTest, controlledGateFires) {
  auto qs = QuantumState({q[0], q[1]}, 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(
      qs.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix(), {q[0]}, {q[0]})
          .succeeded());
  EXPECT_EQ(printed(qs), "|11> -> 1.00");
}

TEST_F(QuantumStateTest, controlledGateDoesNotFire) {
  auto qs = QuantumState({q[0], q[1]}, 4);
  ASSERT_TRUE(
      qs.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix(), {q[0]}, {q[0]})
          .succeeded());
  EXPECT_EQ(printed(qs), "|00> -> 1.00");
}

TEST_F(QuantumStateTest, controlledGateOnSuperposition) {
  auto qs = QuantumState({q[0], q[1]}, 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(
      qs.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix(), {q[0]}, {q[0]})
          .succeeded());
  EXPECT_EQ(printed(qs), "|00> -> 0.71, |11> -> 0.71");
}

TEST_F(QuantumStateTest, appliedGateRenamesControls) {
  auto qs = QuantumState({q[0], q[1]}, 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(
      qs.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix(), {q[0]}, {q[2]})
          .succeeded());
  EXPECT_FALSE(qs.contains(q[0]));
  EXPECT_TRUE(qs.contains(q[2]));
  EXPECT_TRUE(qs.isAlwaysOne(q[1]));
}

TEST_F(QuantumStateTest, controlInOutLengthMismatchFails) {
  auto qs = QuantumState({q[0], q[1], q[2], q[3]}, 4);
  EXPECT_TRUE(
      qs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix(), {q[1]}, {q[2], q[3]})
          .failed());
}

//===----------------------------------------------------------------------===//
// Amplitude budget
//===----------------------------------------------------------------------===//

TEST_F(QuantumStateTest, exceedingAmplitudeBudgetBecomesTop) {
  auto qs = QuantumState({q[0], q[1], q[2], q[3]}, 2);
  ASSERT_TRUE(qs.applyMatrix1Q(q[3], q[3], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(
      qs.applyMatrix1Q(q[2], q[2], xOp.getUnitaryMatrix(), {q[3]}, {q[3]})
          .succeeded());
  EXPECT_FALSE(qs.isTop());
  ASSERT_TRUE(qs.applyMatrix1Q(q[2], q[2], hOp.getUnitaryMatrix()).succeeded());
  EXPECT_TRUE(qs.isTop());
}

TEST_F(QuantumStateTest, topStateStillForwardsQubits) {
  auto qs = QuantumState({q[0], q[1], q[2], q[3]}, 1);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(qs.isTop());
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[1], xOp.getUnitaryMatrix()).succeeded());
  EXPECT_FALSE(qs.contains(q[0]));
  EXPECT_TRUE(qs.contains(q[1]));
}

//===----------------------------------------------------------------------===//
// Controlled phase
//===----------------------------------------------------------------------===//

TEST_F(QuantumStateTest, uncontrolledPhaseIsRejected) {
  auto qs = QuantumState::singletonZero(q[0], 4);
  EXPECT_TRUE(qs.applyControlledPhase(std::acos(-1.0), {}).failed());
}

TEST_F(QuantumStateTest, controlledPhaseAffectsOnlyControlledSubspace) {
  auto qs = QuantumState({q[0], q[1], q[2], q[3]}, 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(qs.applyControlledPhase(std::acos(-1.0), {q[0]}).succeeded());
  EXPECT_EQ(printed(qs), "|0000> -> 0.71, |0001> -> -0.71");
}

TEST_F(QuantumStateTest, controlledPhaseOnQubitNotInGroupFails) {
  auto qs = QuantumState::singletonZero(q[0], 4);
  EXPECT_TRUE(qs.applyControlledPhase(1.0, {q[1]}).failed());
}

//===----------------------------------------------------------------------===//
// Measurement
//===----------------------------------------------------------------------===//

TEST_F(QuantumStateTest, measureDeterministicZero) {
  auto qs = QuantumState::singletonZero(q[0], 2);
  const auto result = qs.measure(q[0], q[0]);
  ASSERT_TRUE(succeeded(result));
  const auto& outcomes = *result;
  ASSERT_EQ(outcomes.size(), 1U);
  EXPECT_EQ(outcomes[0].bit, 0U);
  EXPECT_DOUBLE_EQ(outcomes[0].probability, 1.0);
  EXPECT_TRUE(*outcomes[0].state == qs);
}

TEST_F(QuantumStateTest, measureRenamesMeasuredQubit) {
  auto qs = QuantumState::singletonZero(q[0], 2);
  const auto result = qs.measure(q[0], q[1]);
  ASSERT_TRUE(succeeded(result));
  ASSERT_EQ(result->size(), 1U);
  EXPECT_FALSE(result->front().state->contains(q[0]));
  EXPECT_TRUE(result->front().state->contains(q[1]));
}

TEST_F(QuantumStateTest, measureDeterministicOne) {
  auto qs = QuantumState::singletonZero(q[0], 2);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  const auto result = qs.measure(q[0], q[0]);
  ASSERT_TRUE(succeeded(result));
  const auto& outcomes = *result;
  ASSERT_EQ(outcomes.size(), 1U);
  EXPECT_EQ(outcomes[0].bit, 1U);
  EXPECT_DOUBLE_EQ(outcomes[0].probability, 1.0);
  EXPECT_TRUE(*outcomes[0].state == qs);
}

TEST_F(QuantumStateTest, measureSuperpositionSplits) {
  auto qs = QuantumState({q[0], q[1]}, 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(
      qs.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix(), {q[0]}, {q[0]})
          .succeeded());
  const auto result = qs.measure(q[0], q[0]);
  ASSERT_TRUE(succeeded(result));
  const auto& outcomes = *result;
  ASSERT_EQ(outcomes.size(), 2U);
  EXPECT_EQ(outcomes[0].bit, 0U);
  EXPECT_DOUBLE_EQ(outcomes[0].probability, 0.5);
  EXPECT_EQ(printed(*outcomes[0].state), "|00> -> 1.00");
  EXPECT_EQ(outcomes[1].bit, 1U);
  EXPECT_DOUBLE_EQ(outcomes[1].probability, 0.5);
  EXPECT_EQ(printed(*outcomes[1].state), "|11> -> 1.00");
}

TEST_F(QuantumStateTest, measureQubitNotInGroupFails) {
  auto qs = QuantumState::singletonZero(q[0], 2);
  EXPECT_TRUE(failed(qs.measure(q[1], q[1])));
}

TEST_F(QuantumStateTest, measureOnTopStateYieldsNoBranches) {
  auto qs = QuantumState({q[0], q[1], q[2], q[3]}, 1);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(qs.isTop());
  const auto result = qs.measure(q[0], q[0]);
  ASSERT_TRUE(succeeded(result));
  EXPECT_TRUE(result->empty());
}

//===----------------------------------------------------------------------===//
// Reset
//===----------------------------------------------------------------------===//

TEST_F(QuantumStateTest, resetDeterministicZero) {
  auto qs = QuantumState::singletonZero(q[0], 2);
  const auto result = qs.reset(q[0], q[1]);
  ASSERT_TRUE(succeeded(result));
  const auto& outcomes = *result;
  ASSERT_EQ(outcomes.size(), 1U);
  EXPECT_EQ(outcomes[0].bit, 0U);
  EXPECT_EQ(printed(*outcomes[0].state), "|0> -> 1.00");
  EXPECT_FALSE(outcomes[0].state->contains(q[0]));
  EXPECT_TRUE(outcomes[0].state->contains(q[1]));
}

TEST_F(QuantumStateTest, resetDeterministicOneForcesZero) {
  auto qs = QuantumState::singletonZero(q[0], 2);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  const auto result = qs.reset(q[0], q[0]);
  ASSERT_TRUE(succeeded(result));
  const auto& outcomes = *result;
  ASSERT_EQ(outcomes.size(), 1U);
  EXPECT_EQ(outcomes[0].bit, 1U);
  EXPECT_EQ(printed(*outcomes[0].state), "|0> -> 1.00");
}

TEST_F(QuantumStateTest, resetSuperpositionForcesTargetToZero) {
  auto qs = QuantumState({q[0], q[1]}, 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(
      qs.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix(), {q[0]}, {q[0]})
          .succeeded());
  const auto result = qs.reset(q[0], q[0]);
  ASSERT_TRUE(succeeded(result));
  const auto& outcomes = *result;
  ASSERT_EQ(outcomes.size(), 2U);
  EXPECT_EQ(printed(*outcomes[0].state), "|00> -> 1.00");
  EXPECT_DOUBLE_EQ(outcomes[1].probability, 0.5);
  EXPECT_EQ(printed(*outcomes[1].state), "|10> -> 1.00");
}

TEST_F(QuantumStateTest, resetQubitNotInGroupFails) {
  auto qs = QuantumState::singletonZero(q[0], 2);
  EXPECT_TRUE(failed(qs.reset(q[1], q[1])));
}

TEST_F(QuantumStateTest, resetOnTopStateYieldsNoBranches) {
  auto qs = QuantumState({q[0], q[1], q[2], q[3]}, 1);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(qs.isTop());
  const auto result = qs.reset(q[0], q[0]);
  ASSERT_TRUE(succeeded(result));
  EXPECT_TRUE(result->empty());
}

//===----------------------------------------------------------------------===//
// unify
//===----------------------------------------------------------------------===//

TEST_F(QuantumStateTest, unifyTensorsTwoGroups) {
  auto a = QuantumState::singletonZero(q[0], 10);
  ASSERT_TRUE(a.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  const auto b = QuantumState::singletonZero(q[1], 10);
  const auto unified = a.unify(b);
  EXPECT_EQ(unified.getQubits().size(), 2U);
  EXPECT_EQ(printed(unified), "|00> -> 0.71, |01> -> 0.71");
}

TEST_F(QuantumStateTest, unifyExceedingBudgetIsTop) {
  auto a = QuantumState({q[0], q[1]}, 3);
  ASSERT_TRUE(a.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  auto b = QuantumState({q[2], q[3]}, 3);
  ASSERT_TRUE(b.applyMatrix1Q(q[2], q[2], hOp.getUnitaryMatrix()).succeeded());
  EXPECT_FALSE(a.isTop());
  EXPECT_FALSE(b.isTop());
  EXPECT_TRUE(a.unify(b).isTop());
}

//===----------------------------------------------------------------------===//
// Queries
//===----------------------------------------------------------------------===//

TEST_F(QuantumStateTest, alwaysZeroAndAlwaysOne) {
  auto qs = QuantumState({q[0], q[1]}, 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], xOp.getUnitaryMatrix()).succeeded());
  EXPECT_TRUE(qs.isAlwaysOne(q[0]));
  EXPECT_FALSE(qs.isAlwaysZero(q[0]));
  EXPECT_TRUE(qs.isAlwaysZero(q[1]));
  EXPECT_FALSE(qs.isAlwaysOne(q[1]));

  ASSERT_TRUE(qs.applyMatrix1Q(q[1], q[1], hOp.getUnitaryMatrix()).succeeded());
  EXPECT_FALSE(qs.isAlwaysZero(q[1]));
  EXPECT_FALSE(qs.isAlwaysOne(q[1]));
}

TEST_F(QuantumStateTest, hasAlwaysZeroAmplitude) {
  auto qs = QuantumState({q[0], q[1]}, 4);
  ASSERT_TRUE(qs.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(
      qs.applyMatrix1Q(q[1], q[1], xOp.getUnitaryMatrix(), {q[0]}, {q[0]})
          .succeeded());
  EXPECT_TRUE(qs.hasAlwaysZeroAmplitude({{q[0], false}, {q[1], true}}));
  EXPECT_FALSE(qs.hasAlwaysZeroAmplitude({{q[0], true}, {q[1], true}}));
}

//===----------------------------------------------------------------------===//
// Equality
//===----------------------------------------------------------------------===//

TEST_F(QuantumStateTest, equalityIgnoresNegligibleDifferences) {
  auto a = QuantumState({q[0], q[1], q[2], q[3]}, 4);
  const auto b = QuantumState({q[0], q[1], q[2], q[3]}, 4);
  ASSERT_TRUE(a.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  ASSERT_TRUE(a.applyMatrix1Q(q[0], q[0], hOp.getUnitaryMatrix()).succeeded());
  EXPECT_TRUE(a == b);
}

TEST_F(QuantumStateTest, topStatesAreEqual) {
  auto a = QuantumState({q[0], q[1], q[2], q[3]}, 1);
  auto b = QuantumState({q[0], q[1], q[2], q[3]}, 1);
  a.markTop();
  b.markTop();
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == QuantumState({q[0], q[1], q[2], q[3]}, 4));
}

} // namespace
