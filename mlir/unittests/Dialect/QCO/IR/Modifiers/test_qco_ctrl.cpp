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
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>
#include <string>

using namespace mlir;
using namespace mlir::qco;

class QCOCtrlOpTest : public ::testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder;

  QCOCtrlOpTest() : builder(&context) {}

  void SetUp() override {
    context.loadDialect<QCODialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();

    // Setup Builder
    builder.initialize();
  }

  OwningOpRef<ModuleOp> testParse(const StringRef ctrlOpAssembly) {
    // Wrap the op in a function to provide operands
    const std::string source =
        (Twine("func.func @test(%q0: !qco.qubit, %q1: !qco.qubit) {\n") +
         ctrlOpAssembly + "\n" + "  return\n" + "}")
            .str();
    const ScopedDiagnosticHandler diagHandler(&context);
    return parseSourceString<ModuleOp>(source, &context);
  };
};

TEST_F(QCOCtrlOpTest, LambdaBuilder) {
  // Allocate qubits to use as operands
  const auto q = builder.allocQubitRegister(3);

  // Create CtrlOp using the lambda builder
  builder.ctrl(q[0], {q[1], q[2]},
               [&](ValueRange innerTargets) -> SmallVector<Value> {
                 // Create the inner operation
                 auto [q0, q1] = builder.swap(innerTargets[0], innerTargets[1]);
                 return {q0, q1};
               });
  auto ctrlOp = cast<CtrlOp>(builder.getBlock()->getOperations().back());
  auto module = builder.finalize();

  // Verify the operation structure
  EXPECT_EQ(ctrlOp.getNumControls(), 1);
  EXPECT_EQ(ctrlOp.getNumTargets(), 2);
  EXPECT_EQ(ctrlOp.getResults().size(), 3); // 1 control out + 2 targets out

  // Verify body region
  auto& region = ctrlOp.getRegion();
  ASSERT_FALSE(region.empty());
  auto& block = region.front();
  ASSERT_EQ(block.getOperations().size(), 2); // Body Unitary + Yield

  EXPECT_TRUE(isa<SWAPOp>(block.front()));
  EXPECT_TRUE(isa<YieldOp>(block.back()));

  // Verify target aliasing via block arguments
  const auto qType = QubitType::get(&context);
  ASSERT_EQ(block.getNumArguments(), 2); // 2 target block args
  EXPECT_EQ(block.getArgument(0).getType(), qType);
  EXPECT_EQ(block.getArgument(1).getType(), qType);

  // Verify the SWAP uses block arguments, not original operands
  auto swapOp = cast<SWAPOp>(block.front());
  EXPECT_EQ(swapOp.getOperand(0), block.getArgument(0));
  EXPECT_EQ(swapOp.getOperand(1), block.getArgument(1));

  ASSERT_TRUE(mlir::verify(ctrlOp).succeeded());
}

TEST_F(QCOCtrlOpTest, UnitaryOpBuilder) {
  // Allocate qubits
  const auto q = builder.allocQubitRegister(2);
  const auto qType = QubitType::get(&context);

  // Create a template unitary operation (X gate)
  auto xOp = XOp::create(builder, builder.getUnknownLoc(), q[1]);

  // Create CtrlOp using the UnitaryOpInterface builder
  auto ctrlOp = CtrlOp::create(builder, builder.getUnknownLoc(), q[0], q[1],
                               cast<UnitaryOpInterface>(xOp.getOperation()));

  // Verify structure
  EXPECT_EQ(ctrlOp.getNumControls(), 1);
  EXPECT_EQ(ctrlOp.getNumTargets(), 1);
  EXPECT_EQ(ctrlOp.getResults().size(), 2); // 1 control out + 1 target out

  // Verify body
  auto& region = ctrlOp.getRegion();
  ASSERT_FALSE(region.empty());
  auto& block = region.front();
  ASSERT_EQ(block.getOperations().size(), 2);

  EXPECT_TRUE(isa<XOp>(block.front()));
  EXPECT_TRUE(isa<YieldOp>(block.back()));

  // Verify target aliasing via block arguments
  ASSERT_EQ(block.getNumArguments(), 1); // 1 target block arg
  EXPECT_EQ(block.getArgument(0).getType(), qType);

  // Verify the XOp inside region uses block argument
  auto innerXOp = cast<XOp>(block.front());
  EXPECT_EQ(innerXOp.getOperand(), block.getArgument(0));

  // The template op 'xOp' still exists in the main block before ctrlOp.
  EXPECT_TRUE(mlir::verify(ctrlOp).succeeded());
}

TEST_F(QCOCtrlOpTest, VerifierBodySize) {
  const auto q = builder.allocQubitRegister(2);

  // Create valid CtrlOp
  builder.ctrl(q[0], q[1], [&](ValueRange innerTargets) -> SmallVector<Value> {
    return {builder.x(innerTargets[0])};
  });
  auto ctrlOp = cast<CtrlOp>(builder.getBlock()->getOperations().back());
  auto module = builder.finalize();

  // Insert an extra operation into the body
  auto& region = ctrlOp.getRegion();
  auto& block = region.front();

  const OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(&block.back()); // Before Yield
  // We can insert another XOp
  builder.create<XOp>(builder.getUnknownLoc(), block.getArgument(0));

  // Should fail because body must have exactly 2 operations
  EXPECT_TRUE(mlir::verify(ctrlOp).failed());
}

TEST_F(QCOCtrlOpTest, VerifierBlockArgsCount) {
  const auto q = builder.allocQubitRegister(2);

  // Create valid CtrlOp
  builder.ctrl(q[0], q[1], [&](ValueRange innerTargets) -> SmallVector<Value> {
    return {builder.x(innerTargets[0])};
  });
  auto ctrlOp = cast<CtrlOp>(builder.getBlock()->getOperations().back());
  auto module = builder.finalize();

  // Add an extra argument to the block
  auto& region = ctrlOp.getRegion();
  auto& block = region.front();
  const auto qType = QubitType::get(&context);
  block.addArgument(qType, builder.getUnknownLoc());

  // Should fail because number of block args must match number of targets (1)
  EXPECT_TRUE(mlir::verify(ctrlOp).failed());
}

TEST_F(QCOCtrlOpTest, VerifierInputTypes) {
  const auto q = builder.allocQubitRegister(2);

  // Create valid CtrlOp
  builder.ctrl(q[0], q[1], [&](ValueRange innerTargets) -> SmallVector<Value> {
    return {builder.x(innerTargets[0])};
  });
  auto ctrlOp = cast<CtrlOp>(builder.getBlock()->getOperations().back());
  auto module = builder.finalize();

  // Change the block argument type to a non-qubit
  auto& region = ctrlOp.getRegion();
  auto& block = region.front();
  block.getArgument(0).setType(builder.getI1Type());

  EXPECT_TRUE(mlir::verify(ctrlOp).failed());
}

TEST_F(QCOCtrlOpTest, VerifierBodyFirstOp) {
  const auto q = builder.allocQubitRegister(2);

  // Create valid CtrlOp
  builder.ctrl(q[0], q[1], [&](ValueRange innerTargets) -> SmallVector<Value> {
    return {builder.reset(innerTargets[0])};
  });
  auto ctrlOp = cast<CtrlOp>(builder.getBlock()->getOperations().back());
  auto module = builder.finalize();

  // Should fail because body must use a unitary as first operation
  EXPECT_TRUE(mlir::verify(ctrlOp).failed());
}

TEST_F(QCOCtrlOpTest, ParserErrors) {
  // 1. Missing opening parenthesis for targets
  EXPECT_EQ(
      testParse(
          "qco.ctrl(%q0) targets %a = %q1) { qco.yield %a } : ({!qco.qubit}, "
          "{!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})")
          .get(),
      nullptr);

  // 2. Missing argument name
  EXPECT_EQ(
      testParse(
          "qco.ctrl(%q0) targets ( = %q1) { qco.yield %q1 } : ({!qco.qubit}, "
          "{!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})")
          .get(),
      nullptr);

  // 3. Missing equals sign
  EXPECT_EQ(
      testParse(
          "qco.ctrl(%q0) targets (%a %q1) { qco.yield %a } : ({!qco.qubit}, "
          "{!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})")
          .get(),
      nullptr);

  // 4. Missing operand (old value)
  EXPECT_EQ(
      testParse(
          "qco.ctrl(%q0) targets (%a = ) { qco.yield %a } : ({!qco.qubit}, "
          "{!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})")
          .get(),
      nullptr);

  // 5. Missing closing parenthesis
  EXPECT_EQ(
      testParse(
          "qco.ctrl(%q0) targets (%a = %q1 { qco.yield %a } : ({!qco.qubit}, "
          "{!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})")
          .get(),
      nullptr);
}
