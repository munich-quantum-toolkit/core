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

  const auto q00 = builder.allocQubit();
  const auto q10 = builder.allocQubit();

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