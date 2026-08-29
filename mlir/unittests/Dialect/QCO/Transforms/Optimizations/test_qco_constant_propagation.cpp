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
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>

namespace {

using namespace mlir;
using namespace mlir::qco;

/// Every qco.ctrl in the module, in walk order.
SmallVector<CtrlOp> ctrlOps(ModuleOp module) {
  SmallVector<CtrlOp> ops;
  module.walk([&](const CtrlOp op) { ops.push_back(op); });
  return ops;
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
  // reg[0] stays |0>, so the controlled X can never fire.
  builder.cx(reg[0], reg[1]);
  const auto module = builder.finalize();

  ASSERT_TRUE(succeeded(run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));
  EXPECT_TRUE(ctrlOps(*module).empty());
}

TEST_F(ConstantPropagationTest, stripsAlwaysSatisfiedControl) {
  auto reg = builder.allocQubitRegister(3);
  const Value one = builder.x(reg[0]);
  const Value sup = builder.h(reg[2]);
  const SmallVector<Value> controls{one, sup};
  builder.mcx(controls, reg[1]);
  const auto module = builder.finalize();

  ASSERT_TRUE(succeeded(run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));

  auto ctrls = ctrlOps(*module);
  ASSERT_EQ(ctrls.size(), 1U);
  EXPECT_EQ(ctrls.front().getNumControls(), 1U);
}

TEST_F(ConstantPropagationTest, keepsGateWhenEveryControlRedundant) {
  auto reg = builder.allocQubitRegister(2);
  const Value one = builder.x(reg[0]); // always |1>
  (void)builder.cx(one, reg[1]);
  const auto module = builder.finalize();

  ASSERT_TRUE(succeeded(run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));

  // Unwrapping a fully-redundant control to an uncontrolled gate is out of
  // v2.0 scope: the op is left untouched.
  auto ctrls = ctrlOps(*module);
  ASSERT_EQ(ctrls.size(), 1U);
  EXPECT_EQ(ctrls.front().getNumControls(), 1U);
}

TEST_F(ConstantPropagationTest, leavesGateAloneWhenStateIsImprecise) {
  auto reg = builder.allocQubitRegister(2);
  const Value sup = builder.h(reg[0]);
  builder.cx(sup, reg[1]);
  const auto module = builder.finalize();

  // Budget of one amplitude forces reg[0] to top before the controlled gate.
  ASSERT_TRUE(succeeded(run(*module, 1)));
  EXPECT_TRUE(succeeded(verify(*module)));
  EXPECT_EQ(ctrlOps(*module).size(), 1U);
}

} // namespace
