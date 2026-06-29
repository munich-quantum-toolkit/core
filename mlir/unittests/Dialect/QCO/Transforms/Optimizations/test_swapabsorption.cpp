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
#include "mlir/Support/IRVerification.h"

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
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

#include <cassert>
#include <memory>

using namespace mlir;
using namespace mlir::qco;

namespace {

class SwapAbsorbPassTest : public testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder programBuilder;
  QCOProgramBuilder referenceBuilder;
  OwningOpRef<ModuleOp> swapModule;
  OwningOpRef<ModuleOp> reference;

  SwapAbsorbPassTest() : programBuilder(&context), referenceBuilder(&context) {}

protected:
  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    programBuilder.initialize();
    referenceBuilder.initialize();
  }

  static LogicalResult applySwapAbsorbPass(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(qco::createSwapAbsorption());
    return pm.run(module);
  }

  static LogicalResult applyCanonicalizerPass(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(createCanonicalizerPass());
    return pm.run(module);
  }
};
}; // namespace

TEST_F(SwapAbsorbPassTest, PassReordersTwoQubitCircuitWithLeadingSwap) {
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  std::tie(q0, q1) = programBuilder.swap(q0, q1);
  q0 = programBuilder.x(q0);
  q1 = programBuilder.h(q1);
  swapModule = programBuilder.finalize();

  auto qRef0 = referenceBuilder.allocQubit();
  auto qRef1 = referenceBuilder.allocQubit();
  qRef0 = referenceBuilder.h(qRef0);
  qRef1 = referenceBuilder.x(qRef1);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(applySwapAbsorbPass(swapModule.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(swapModule.get(), reference.get()));
}

TEST_F(SwapAbsorbPassTest,
       PassReordersTwoQubitCircuitWithSwapAfterSingleQubitGate) {
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  q0 = programBuilder.x(q0);
  q1 = programBuilder.y(q1);
  std::tie(q0, q1) = programBuilder.swap(q0, q1);
  q0 = programBuilder.z(q0);
  q1 = programBuilder.h(q1);
  swapModule = programBuilder.finalize();

  auto qRef0 = referenceBuilder.allocQubit();
  auto qRef1 = referenceBuilder.allocQubit();
  qRef0 = referenceBuilder.x(qRef0);
  qRef1 = referenceBuilder.y(qRef1);
  qRef0 = referenceBuilder.h(qRef0);
  qRef1 = referenceBuilder.z(qRef1);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(applySwapAbsorbPass(swapModule.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(swapModule.get(), reference.get()));
}

TEST_F(SwapAbsorbPassTest, PassReordersCircuitWithMultipleLeadingSwap) {
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();
  auto q3 = programBuilder.allocQubit();
  std::tie(q0, q1) = programBuilder.swap(q0, q1);
  std::tie(q2, q3) = programBuilder.swap(q2, q3);
  q0 = programBuilder.x(q0);
  q1 = programBuilder.h(q1);
  q2 = programBuilder.h(q2);
  q3 = programBuilder.x(q3);
  swapModule = programBuilder.finalize();

  auto qRef0 = referenceBuilder.allocQubit();
  auto qRef1 = referenceBuilder.allocQubit();
  auto qRef2 = referenceBuilder.allocQubit();
  auto qRef3 = referenceBuilder.allocQubit();
  qRef0 = referenceBuilder.h(qRef0);
  qRef1 = referenceBuilder.x(qRef1);
  qRef2 = referenceBuilder.x(qRef2);
  qRef3 = referenceBuilder.h(qRef3);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(applySwapAbsorbPass(swapModule.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(swapModule.get(), reference.get()));
}

TEST_F(SwapAbsorbPassTest, PassReordersCircuitWithDependentSwap) {
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();
  std::tie(q0, q1) = programBuilder.swap(q0, q1);
  std::tie(q1, q2) = programBuilder.swap(q1, q2);
  q0 = programBuilder.x(q0);
  q1 = programBuilder.y(q1);
  q2 = programBuilder.z(q2);
  swapModule = programBuilder.finalize();

  auto qRef00 = referenceBuilder.allocQubit();
  auto qRef10 = referenceBuilder.allocQubit();
  auto qRef20 = referenceBuilder.allocQubit();
  auto qRef01 = referenceBuilder.x(qRef10);
  auto qRef11 = referenceBuilder.y(qRef20);
  auto qRef21 = referenceBuilder.z(qRef00);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(applySwapAbsorbPass(swapModule.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(swapModule.get(), reference.get()));
}
