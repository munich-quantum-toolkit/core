/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
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
  OpBuilder builder;
  OwningOpRef<ModuleOp> module;

  QCOCtrlOpTest() : builder(&context) {}

  void SetUp() override {
    context.loadDialect<QCODialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();

    // Setup Module and Function
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());

    auto funcType = builder.getFunctionType({}, {});
    auto func =
        builder.create<func::FuncOp>(builder.getUnknownLoc(), "main", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(entryBlock);
  }

  bool testParse(StringRef ctrlOpAssembly) {
    // Wrap the op in a function to provide operands
    const std::string source =
        (Twine("func.func @test(%q0: !qco.qubit, %q1: !qco.qubit) {\n") +
         ctrlOpAssembly + "\n" + "  return\n" + "}")
            .str();
    const ScopedDiagnosticHandler diagHandler(&context);
    // Parse should fail
    return parseSourceString<ModuleOp>(source, &context).get() == nullptr;
  };
};

TEST_F(QCOCtrlOpTest, LambdaBuilder) {
  // Allocate qubits to use as operands
  auto qType = QubitType::get(&context);
  auto q0 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();
  auto q1 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();
  auto q2 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();

  const SmallVector<Value> controls = {q0};
  const SmallVector<Value> targets = {q1, q2};

  // Create CtrlOp using the lambda builder
  auto ctrlOp = builder.create<CtrlOp>(
      builder.getUnknownLoc(), controls, targets,
      [&](ValueRange innerTargets) -> ValueRange {
        // Create the inner operation (e.g. SwapOp on the two targets)
        auto swapOp = builder.create<SWAPOp>(builder.getUnknownLoc(),
                                             innerTargets[0], innerTargets[1]);
        return swapOp.getResults();
      });

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
  auto qType = QubitType::get(&context);
  auto q0 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();
  auto q1 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();

  const SmallVector<Value> controls = {q0};
  const SmallVector<Value> targets = {q1};

  // Create a template unitary operation (X gate)
  auto xOp = builder.create<XOp>(builder.getUnknownLoc(), q1);

  // Create CtrlOp using the UnitaryOpInterface builder
  auto ctrlOp =
      builder.create<CtrlOp>(builder.getUnknownLoc(), controls, targets,
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
  auto q0 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();
  auto q1 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();

  // Create valid CtrlOp
  auto ctrlOp = builder.create<CtrlOp>(
      builder.getUnknownLoc(), ValueRange{q0}, ValueRange{q1},
      [&](ValueRange innerTargets) -> ValueRange {
        auto xOp =
            builder.create<XOp>(builder.getUnknownLoc(), innerTargets[0]);
        return xOp->getResults();
      });

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
  auto qType = QubitType::get(&context);
  auto q0 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();
  auto q1 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();

  auto ctrlOp = builder.create<CtrlOp>(
      builder.getUnknownLoc(), ValueRange{q0}, ValueRange{q1},
      [&](ValueRange innerTargets) -> ValueRange {
        auto xOp =
            builder.create<XOp>(builder.getUnknownLoc(), innerTargets[0]);
        return xOp->getResults();
      });

  // Add an extra argument to the block
  auto& region = ctrlOp.getRegion();
  auto& block = region.front();
  block.addArgument(qType, builder.getUnknownLoc());

  // Should fail because number of block args must match number of targets (1)
  EXPECT_TRUE(mlir::verify(ctrlOp).failed());
}

TEST_F(QCOCtrlOpTest, VerifierInputTypes) {
  auto q0 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();
  auto q1 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();

  const SmallVector<Value> controls = {q0};
  const SmallVector<Value> targets = {q1};

  // Create a CtrlOp using a qubit as target.
  auto ctrlOp =
      builder.create<CtrlOp>(builder.getUnknownLoc(), controls, targets,
                             [&](ValueRange innerTargets) -> ValueRange {
                               auto xOp = builder.create<XOp>(
                                   builder.getUnknownLoc(), innerTargets[0]);
                               return xOp->getResults();
                             });

  // Change the block argument type to a non-qubit
  auto& region = ctrlOp.getRegion();
  auto& block = region.front();
  block.getArgument(0).setType(builder.getI1Type());

  EXPECT_TRUE(mlir::verify(ctrlOp).failed());
}

TEST_F(QCOCtrlOpTest, VerifierBodyFirstOp) {
  auto q0 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();
  auto q1 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();

  // Create CtrlOp that uses a non-unitary as first operation in body
  auto ctrlOp = builder.create<CtrlOp>(
      builder.getUnknownLoc(), ValueRange{q0}, ValueRange{q1},
      [&](ValueRange innerTargets) -> ValueRange {
        auto resetOp =
            builder.create<ResetOp>(builder.getUnknownLoc(), innerTargets[0]);
        return resetOp->getResults();
      });

  // Should fail because body must use a unitary as first operation
  EXPECT_TRUE(mlir::verify(ctrlOp).failed());
}

TEST_F(QCOCtrlOpTest, ParserErrors) {
  // 1. Missing opening parenthesis for targets
  EXPECT_TRUE(testParse(
      "qco.ctrl(%q0) targets %a = %q1) { qco.yield %a } : ({!qco.qubit}, "
      "{!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})"));

  // 2. Missing argument name
  EXPECT_TRUE(testParse(
      "qco.ctrl(%q0) targets ( = %q1) { qco.yield %q1 } : ({!qco.qubit}, "
      "{!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})"));

  // 3. Missing equals sign
  EXPECT_TRUE(testParse(
      "qco.ctrl(%q0) targets (%a %q1) { qco.yield %a } : ({!qco.qubit}, "
      "{!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})"));

  // 4. Missing operand (old value)
  EXPECT_TRUE(testParse(
      "qco.ctrl(%q0) targets (%a = ) { qco.yield %a } : ({!qco.qubit}, "
      "{!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})"));

  // 5. Missing closing parenthesis
  EXPECT_TRUE(testParse(
      "qco.ctrl(%q0) targets (%a = %q1 { qco.yield %a } : ({!qco.qubit}, "
      "{!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})"));
}
