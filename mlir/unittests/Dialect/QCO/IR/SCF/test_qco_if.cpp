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

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;
using namespace mlir::qco;

class QCOIfOpTest : public ::testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder;
  OwningOpRef<ModuleOp> module;

  QCOIfOpTest() : builder(&context) {}

  void SetUp() override {
    context.loadDialect<QCODialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();

    // Setup Builder
    builder.initialize();
  }

  /**
   * @brief Counts the amount of operations the current module/circuit
   *        contains.
   */
  template <typename OpTy> int countOps() {
    int count = 0;
    module->walk([&](OpTy) { ++count; });
    return count;
  }
  /**
   * @brief Build a basic qco program with 2 qubits and an if operation with QCO
   * builder.
   */
  IfOp buildOp() {
    const auto q = builder.allocQubitRegister(2);
    auto [qubit, measureResult] = builder.measure(q[0]);

    builder.qcoIf(measureResult, {qubit, q[1]},
                  [&](ValueRange args) -> SmallVector<Value> {
                    auto q2 = builder.h(args[0]);
                    return {q2, args[1]};
                  });
    auto ifOp = cast<IfOp>(builder.getBlock()->getOperations().back());
    module = builder.finalize();
    return ifOp;
  }

  /**
   * @brief Run the canonicalizer on the module and return if it succeeded.
   */
  bool canonicalize() {
    PassManager pm(&context);
    auto* moduleOp = module->getOperation();
    pm.addPass(createCanonicalizerPass());
    return pm.run(moduleOp).succeeded();
  }
};

TEST_F(QCOIfOpTest, TestQCOIfBuilder) {
  // Test If construction with QCO builder
  auto ifOp = buildOp();

  // Verify the operation structure
  EXPECT_EQ(ifOp.getQubits().size(), 2);
  EXPECT_EQ(ifOp.getResults().size(), 2);
  EXPECT_EQ(ifOp.thenYield()->getNumOperands(), 2);
  EXPECT_EQ(ifOp.elseYield()->getNumOperands(), 2);
  EXPECT_EQ(ifOp.thenBlock()->getArgumentTypes(),
            ifOp.elseBlock()->getArgumentTypes());
  EXPECT_EQ(ifOp->getResultTypes(), ifOp.getQubits().getTypes());
  EXPECT_EQ(ifOp->getResultTypes(), ifOp.thenBlock()->getArgumentTypes());

  // Verify operation
  ASSERT_TRUE(verify(ifOp).succeeded());
}

TEST_F(QCOIfOpTest, TestIfBuilder) {
  // Test If construction directly
  const auto q = builder.allocQubit();
  auto constantBool =
      arith::ConstantOp::create(builder, builder.getBoolAttr(true));

  auto ifOp = IfOp::create(
      builder, constantBool, q,
      [&](ValueRange args) -> SmallVector<Value> { return args; },
      [&](ValueRange args) -> SmallVector<Value> { return args; });

  // Verify the operation structure
  EXPECT_EQ(ifOp.getQubits().size(), 1);
  EXPECT_EQ(ifOp.getResults().size(), 1);
  EXPECT_EQ(ifOp.thenYield()->getNumOperands(), 1);
  EXPECT_EQ(ifOp.elseYield()->getNumOperands(), 1);
  EXPECT_EQ(ifOp.thenBlock()->getArgumentTypes(),
            ifOp.elseBlock()->getArgumentTypes());
  EXPECT_EQ(ifOp->getResultTypes(), ifOp.getQubits().getTypes());
  EXPECT_EQ(ifOp->getResultTypes(), ifOp.thenBlock()->getArgumentTypes());

  // Verify operation
  ASSERT_TRUE(verify(ifOp).succeeded());
}

TEST_F(QCOIfOpTest, TestWrongType) {
  auto ifOp = buildOp();

  // Change the block argument type to a non-qubit
  auto* block = ifOp.thenBlock();
  block->getArgument(0).setType(builder.getI1Type());

  // Verify operation
  ASSERT_TRUE(verify(ifOp).failed());
}

TEST_F(QCOIfOpTest, TestSameNumberOfBlockArgs) {
  auto ifOp = buildOp();

  // Add an additional block argument in the then block
  auto* block = ifOp.thenBlock();
  block->addArgument(QubitType::get(&context), builder.getUnknownLoc());

  // Verify operation
  ASSERT_TRUE(mlir::verify(ifOp).failed());
}

TEST_F(QCOIfOpTest, TestSameNumberOfOperandQubitsAndResult) {
  auto ifOp = buildOp();

  // Add an additional block argument in both blocks
  const auto qcoType = QubitType::get(&context);
  auto* thenBlock = ifOp.thenBlock();
  thenBlock->addArgument(qcoType, builder.getUnknownLoc());
  auto* elseBlock = ifOp.elseBlock();
  elseBlock->addArgument(qcoType, builder.getUnknownLoc());

  // Verify operation
  ASSERT_TRUE(verify(ifOp).failed());
}

TEST_F(QCOIfOpTest, TestConstantCondition) {
  // Build a qco.if with a constant condition
  const auto q = builder.allocQubitRegister(2);

  builder.qcoIf(arith::ConstantOp::create(builder, builder.getBoolAttr(true)),
                q, [&](ValueRange args) -> SmallVector<Value> {
                  auto q2 = builder.h(args[0]);
                  return {q2, args[1]};
                });

  module = builder.finalize();

  // Run canonicalizer
  ASSERT_TRUE(canonicalize());

  EXPECT_EQ(countOps<HOp>(), 1);
  EXPECT_EQ(countOps<IfOp>(), 0);
}

TEST_F(QCOIfOpTest, TestConditionPropagation) {
  // Test to check if the condition is propagated into the regions
  const auto q = builder.allocQubitRegister(2);
  auto [qubitOut, measureResult] = builder.measure(q[0]);

  builder.qcoIf(measureResult, {qubitOut, q[1]},
                [&](ValueRange args) -> SmallVector<Value> {
                  auto innerIf = builder.qcoIf(
                      measureResult, args,
                      [&](ValueRange innerArgs) -> SmallVector<Value> {
                        auto q2 = builder.h(innerArgs[0]);
                        return {q2, innerArgs[1]};
                      });
                  return innerIf;
                });

  module = builder.finalize();

  // Run canonicalizer
  ASSERT_TRUE(canonicalize());

  EXPECT_EQ(countOps<HOp>(), 1);
  EXPECT_EQ(countOps<IfOp>(), 1);
}
