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
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>

namespace {

using namespace mlir;
using namespace mlir::qco;

/// Number of ops of a given kind anywhere in the module (bodies included).
template <typename OpT> unsigned countOps(ModuleOp module) {
  unsigned n = 0;
  module.walk([&](OpT) { ++n; });
  return n;
}

/// The first op of a given kind in walk order, or a null handle if there is
/// none.
template <typename OpT> OpT firstOp(ModuleOp module) {
  OpT found;
  module.walk([&](OpT op) {
    found = op;
    return WalkResult::interrupt();
  });
  return found;
}

class ConstantPropagationTest : public testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder;

  ConstantPropagationTest() : builder(&context) {}

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect,
                    qtensor::QTensorDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    builder.initialize();
  }

  static LogicalResult run(ModuleOp module, const std::size_t maxAmplitudes = 4,
                           const std::size_t maxHybridStates = 4) {
    PassManager pm(module.getContext());
    pm.addPass(createConstantPropagation(
        ConstantPropagationOptions{.maximumNonzeroAmplitudes = maxAmplitudes,
                                   .maximumHybridStates = maxHybridStates}));
    return pm.run(module);
  }
};

TEST_F(ConstantPropagationTest, dropsGateWithUnsatisfiableControl) {
  auto reg = builder.allocQubitRegister(2);
  builder.cx(reg[0], reg[1]);
  const auto module = builder.finalize();

  ASSERT_TRUE(succeeded(run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));
  EXPECT_EQ(countOps<CtrlOp>(*module), 0U);
}

TEST_F(ConstantPropagationTest, stripsAlwaysSatisfiedControl) {
  auto reg = builder.allocQubitRegister(3);
  Value one = builder.x(reg[0]);
  Value sup = builder.h(reg[2]);
  const SmallVector<Value> controls{one, sup};
  builder.mcx(controls, reg[1]);
  const auto module = builder.finalize();

  ASSERT_TRUE(succeeded(run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));

  ASSERT_EQ(countOps<CtrlOp>(*module), 1U);
  EXPECT_EQ(firstOp<CtrlOp>(*module).getNumControls(), 1U);
}

TEST_F(ConstantPropagationTest, unwrapsGateWhenEveryControlRedundant) {
  auto reg = builder.allocQubitRegister(2);
  Value one = builder.x(reg[0]);
  builder.cx(one, reg[1]);
  const auto module = builder.finalize();

  ASSERT_TRUE(succeeded(run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));

  EXPECT_EQ(countOps<CtrlOp>(*module), 0U);
  EXPECT_EQ(countOps<XOp>(*module), 2U);
}

TEST_F(ConstantPropagationTest, leavesGateAloneWhenStateIsImprecise) {
  auto reg = builder.allocQubitRegister(2);
  Value sup = builder.h(reg[0]);
  builder.cx(sup, reg[1]);
  const auto module = builder.finalize();

  ASSERT_TRUE(succeeded(run(*module, 1)));
  EXPECT_TRUE(succeeded(verify(*module)));
  EXPECT_EQ(countOps<CtrlOp>(*module), 1U);
}

TEST_F(ConstantPropagationTest, dropsGateWhenOneOfSeveralControlsIsAlwaysZero) {
  auto reg = builder.allocQubitRegister(3);
  Value one = builder.x(reg[0]);
  const SmallVector<Value> controls{one, reg[1]};
  builder.mcx(controls, reg[2]);
  const auto module = builder.finalize();

  ASSERT_TRUE(succeeded(run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));
  EXPECT_EQ(countOps<CtrlOp>(*module), 0U);
  EXPECT_EQ(countOps<XOp>(*module), 1U);
}

TEST_F(ConstantPropagationTest, stripsTrailingAlwaysOneControl) {
  auto reg = builder.allocQubitRegister(3);
  Value sup = builder.h(reg[0]);
  Value one = builder.x(reg[1]);
  const SmallVector<Value> controls{sup, one};
  builder.mcx(controls, reg[2]);
  const auto module = builder.finalize();

  ASSERT_TRUE(succeeded(run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));

  ASSERT_EQ(countOps<CtrlOp>(*module), 1U);
  auto ctrl = firstOp<CtrlOp>(*module);
  EXPECT_EQ(ctrl.getNumControls(), 1U);
  EXPECT_TRUE(ctrl.getInputControl(0) == sup);
  EXPECT_EQ(countOps<XOp>(*module), 2U);
}

TEST_F(ConstantPropagationTest, stripsAllButOneControl) {
  auto reg = builder.allocQubitRegister(4);
  Value a = builder.x(reg[0]);
  Value b = builder.x(reg[1]);
  Value sup = builder.h(reg[3]);
  const SmallVector<Value> controls{a, b, sup};
  builder.mcx(controls, reg[2]);
  const auto module = builder.finalize();

  ASSERT_TRUE(succeeded(run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));

  ASSERT_EQ(countOps<CtrlOp>(*module), 1U);
  auto ctrl = firstOp<CtrlOp>(*module);
  EXPECT_EQ(ctrl.getNumControls(), 1U);
  EXPECT_TRUE(ctrl.getInputControl(0) == sup);
}

TEST_F(ConstantPropagationTest,
       unwrapsMultiControlGateWhenAllControlsRedundant) {
  auto reg = builder.allocQubitRegister(3);
  Value a = builder.x(reg[0]);
  Value b = builder.x(reg[1]);
  const SmallVector<Value> controls{a, b};
  builder.mcx(controls, reg[2]);
  const auto module = builder.finalize();

  ASSERT_TRUE(succeeded(run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));

  EXPECT_EQ(countOps<CtrlOp>(*module), 0U);
  EXPECT_EQ(countOps<XOp>(*module), 3U);
}

TEST_F(ConstantPropagationTest, simplifiesChainOfControlledGates) {
  auto reg = builder.allocQubitRegister(3);
  Value q0 = builder.x(reg[0]);
  Value q1 = builder.cx(q0, reg[1]).second;
  builder.cx(q1, reg[2]);
  const auto module = builder.finalize();

  ASSERT_TRUE(succeeded(run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));

  EXPECT_EQ(countOps<CtrlOp>(*module), 0U);
  EXPECT_EQ(countOps<XOp>(*module), 3U);
}

TEST_F(ConstantPropagationTest, noControlledGatesIsNoOp) {
  auto reg = builder.allocQubitRegister(2);
  (void)builder.x(reg[0]);
  (void)builder.h(reg[1]);
  const auto module = builder.finalize();

  ASSERT_TRUE(succeeded(run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));
  EXPECT_EQ(countOps<XOp>(*module), 1U);
  EXPECT_EQ(countOps<HOp>(*module), 1U);
}

} // namespace
