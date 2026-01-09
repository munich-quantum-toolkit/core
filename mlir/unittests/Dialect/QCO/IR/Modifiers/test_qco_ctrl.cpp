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
#include <mlir/IR/MLIRContext.h>

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
    builder.setInsertionPointToStart(entryBlock);
  }
};

TEST_F(QCOCtrlOpTest, LambdaBuilder) {
  // Allocate qubits to use as operands
  auto q0 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();
  auto q1 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();
  auto q2 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();

  ValueRange controls = {q0};
  ValueRange targets = {q1, q2};

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
  EXPECT_FALSE(region.empty());
  auto& block = region.front();
  EXPECT_EQ(block.getOperations().size(), 2); // Body Unitary + Yield

  EXPECT_TRUE(isa<SWAPOp>(block.front()));
  EXPECT_TRUE(isa<YieldOp>(block.back()));

  EXPECT_TRUE(module->verify().succeeded());
}

TEST_F(QCOCtrlOpTest, UnitaryOpBuilder) {
  // Allocate qubits
  auto q0 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();
  auto q1 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();

  ValueRange controls = {q0};
  ValueRange targets = {q1};

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
  EXPECT_FALSE(region.empty());
  auto& block = region.front();
  EXPECT_EQ(block.getOperations().size(), 2);

  EXPECT_TRUE(isa<XOp>(block.front()));
  EXPECT_TRUE(isa<YieldOp>(block.back()));

  // The template op 'xOp' still exists in the main block before ctrlOp.
  EXPECT_TRUE(module->verify().succeeded());
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

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(&block.back()); // Before Yield
  // We can insert another XOp
  builder.create<XOp>(builder.getUnknownLoc(), block.getArgument(0));

  // Should fail because body must have exactly 2 operations
  EXPECT_TRUE(ctrlOp.verify().failed());
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
  EXPECT_TRUE(ctrlOp.verify().failed());
}

TEST_F(QCOCtrlOpTest, VerifierInputTypes) {
  auto q0 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();
  auto q1 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();

  // Create a non-qubit value (a constant i1)
  auto cstOp =
      builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 1, 1);
  Value nonQubit = cstOp.getResult();

  ValueRange controls = {q0};
  ValueRange targets = {q1};

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
  block.getArgument(0).setType(nonQubit.getType());

  EXPECT_TRUE(ctrlOp.verify().failed());
}

TEST_F(QCOCtrlOpTest, VerifierBodyFirstOp) {
  auto q0 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();
  auto q1 = builder.create<AllocOp>(builder.getUnknownLoc()).getResult();

  // Create valid CtrlOp
  auto ctrlOp = builder.create<CtrlOp>(
      builder.getUnknownLoc(), ValueRange{q0}, ValueRange{q1},
      [&](ValueRange innerTargets) -> ValueRange {
        auto resetOp =
            builder.create<ResetOp>(builder.getUnknownLoc(), innerTargets[0]);
        return resetOp->getResults();
      });

  // Insert an extra operation into the body
  auto& region = ctrlOp.getRegion();
  auto& block = region.front();

  // Should fail because body must have exactly 2 operations
  EXPECT_TRUE(ctrlOp.verify().failed());
}
