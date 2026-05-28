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

#include <gtest/gtest.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <memory>

using namespace mlir;
using namespace mlir::qco;

namespace {

class SwapAbsorbPassTest : public testing::Test {

protected:
  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<qco::QCODialect, arith::ArithDialect, func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  static void applySwapAbsorb(OwningOpRef<ModuleOp>& moduleOp) {
    PassManager pm(moduleOp->getContext());
    pm.addPass(qco::createSwapAbsorptionPass());
    auto res = pm.run(*moduleOp);

    ASSERT_TRUE(succeeded(res));
  }

  std::unique_ptr<MLIRContext> context;
};
}; // namespace

TEST_F(SwapAbsorbPassTest, PassDoesNotChangeSwaplessProgram) {

  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = builder.staticQubit(0);
  const auto q10 = builder.staticQubit(1);

  const auto q01 = builder.h(q00);
  const auto [q02, q11] = builder.cx(q01, q10);

  builder.sink(q02);
  builder.sink(q11);

  auto moduleThroughPass = builder.finalize();
  auto originalModule = moduleThroughPass->clone();

  applySwapAbsorb(moduleThroughPass);
  ASSERT_TRUE(mlir::OperationEquivalence::isEquivalentTo(
      moduleThroughPass.get(), originalModule,
      mlir::OperationEquivalence::Flags::IgnoreLocations));
}

TEST_F(SwapAbsorbPassTest, PassReordersTwoQubitCircuitWithLeadingSwap) {

  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = builder.staticQubit(0);
  const auto q10 = builder.staticQubit(1);

  const auto [q01, q11] = builder.swap(q00, q10);

  const auto q02 = builder.id(q01);
  const auto q12 = builder.id(q11);

  builder.sink(q02);
  builder.sink(q12);

  auto moduleThroughPass = builder.finalize();
  applySwapAbsorb(moduleThroughPass);

  ASSERT_EQ(q10, ((IdOp)q02.getDefiningOp()).getInputQubit(0));
  ASSERT_EQ(q00, ((IdOp)q12.getDefiningOp()).getInputQubit(0));
}

TEST_F(SwapAbsorbPassTest, PassAbsorbsTwoIndependentSwaps) {

  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = builder.staticQubit(0);
  const auto q10 = builder.staticQubit(1);
  const auto q20 = builder.staticQubit(2);
  const auto q30 = builder.staticQubit(3);

  const auto [q01, q11] = builder.swap(q00, q10);
  const auto [q21, q31] = builder.swap(q20, q30);

  const auto q02 = builder.id(q01);
  const auto q12 = builder.id(q11);
  const auto q22 = builder.id(q21);
  const auto q32 = builder.id(q31);

  builder.sink(q02);
  builder.sink(q12);
  builder.sink(q22);
  builder.sink(q32);

  auto moduleThroughPass = builder.finalize();
  applySwapAbsorb(moduleThroughPass);

  ASSERT_EQ(q10, ((IdOp)q02.getDefiningOp()).getInputQubit(0));
  ASSERT_EQ(q00, ((IdOp)q12.getDefiningOp()).getInputQubit(0));
  ASSERT_EQ(q30, ((IdOp)q22.getDefiningOp()).getInputQubit(0));
  ASSERT_EQ(q20, ((IdOp)q32.getDefiningOp()).getInputQubit(0));
}

TEST_F(SwapAbsorbPassTest, PassAbsorbsSwapWithLeadingSingleQubitGates) {

  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = builder.staticQubit(0);
  const auto q10 = builder.staticQubit(1);

  const auto q01 = builder.id(q00);
  const auto q11 = builder.id(q10);

  const auto [q02, q12] = builder.swap(q01, q11);

  const auto q03 = builder.id(q02);
  const auto q13 = builder.id(q12);

  builder.sink(q03);
  builder.sink(q13);

  auto moduleThroughPass = builder.finalize();
  applySwapAbsorb(moduleThroughPass);

  ASSERT_EQ(q11, ((IdOp)q03.getDefiningOp()).getInputQubit(0));
  ASSERT_EQ(q01, ((IdOp)q13.getDefiningOp()).getInputQubit(0));
}

TEST_F(SwapAbsorbPassTest, PassAbsorbsTwoDependentSwaps) {

  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto q00 = builder.staticQubit(0);
  const auto q10 = builder.staticQubit(1);
  const auto q20 = builder.staticQubit(2);

  const auto [q01, q11] = builder.swap(q00, q10);
  const auto [q12, q21] = builder.swap(q11, q20);

  const auto q02 = builder.id(q01);
  const auto q13 = builder.id(q12);
  const auto q22 = builder.id(q21);

  builder.sink(q02);
  builder.sink(q13);
  builder.sink(q22);

  auto moduleThroughPass = builder.finalize();
  applySwapAbsorb(moduleThroughPass);

  ASSERT_EQ(q20, ((IdOp)q13.getDefiningOp()).getInputQubit(0));
  ASSERT_EQ(q00, ((IdOp)q22.getDefiningOp()).getInputQubit(0));
  ASSERT_EQ(q10, ((IdOp)q02.getDefiningOp()).getInputQubit(0));
}
