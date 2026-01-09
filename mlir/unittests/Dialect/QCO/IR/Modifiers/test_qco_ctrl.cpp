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
  auto qType = QubitType::get(&context);
  auto q0 = builder
                .create<AllocOp>(builder.getUnknownLoc(), qType, nullptr,
                                 nullptr, nullptr)
                .getResult();
  auto q1 = builder
                .create<AllocOp>(builder.getUnknownLoc(), qType, nullptr,
                                 nullptr, nullptr)
                .getResult();
  auto q2 = builder
                .create<AllocOp>(builder.getUnknownLoc(), qType, nullptr,
                                 nullptr, nullptr)
                .getResult();

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
  auto qType = QubitType::get(&context);
  auto q0 = builder
                .create<AllocOp>(builder.getUnknownLoc(), qType, nullptr,
                                 nullptr, nullptr)
                .getResult();
  auto q1 = builder
                .create<AllocOp>(builder.getUnknownLoc(), qType, nullptr,
                                 nullptr, nullptr)
                .getResult();

  ValueRange controls = {q0};
  ValueRange targets = {q1};

  // Create a template unitary operation (X gate)
  // We create it in the module mainly to get a handle to it.
  // In a real scenario, this might be a temporary op or one we want to "wrap".
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
  // This is expected behavior for the builder as it clones the op.
  // Just ensuring everything is valid.
  EXPECT_TRUE(module->verify().succeeded());
}
