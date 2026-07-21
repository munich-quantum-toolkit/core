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
#include "mlir/Support/IRVerification.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

#include <tuple>

namespace {

using namespace mlir;
using namespace mlir::qco;

class QCOQubitReuseTest : public testing::Test {

protected:
  MLIRContext context;
  QCOProgramBuilder programBuilder;
  QCOProgramBuilder referenceBuilder;
  OwningOpRef<ModuleOp> module;
  OwningOpRef<ModuleOp> reference;

  QCOQubitReuseTest() : programBuilder(&context), referenceBuilder(&context) {}

  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  /**
   * @brief Adds the replaceClassicalControls pass to the current context and
   * runs it.
   */
  static LogicalResult
  runReplaceClassicalControlsPass(ModuleOp module,
                                  bool liftMeasurements = false) {
    PassManager pm(module.getContext());
    pm.addPass(createReuseQubits());
    if (liftMeasurements) {
      pm.addPass(createMeasurementLifting());
      pm.addPass(createReplaceClassicalControls());
    }
    pm.addPass(createCanonicalizerPass());
    return pm.run(module);
  }

  /**
   * @brief Adds the canonicalizerPass to the current context and runs it.
   */
  static LogicalResult runCanonicalizerPass(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(createCanonicalizerPass());
    return pm.run(module);
  }
};

} // namespace

TEST_F(QCOQubitReuseTest, replaceClassicalControlsOnlyControl) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();

  Value c0;
  Value c1;

  q0 = programBuilder.h(q0);
  q1 = programBuilder.h(q1);

  std::tie(q0, c0) = programBuilder.measure(q0);
  std::tie(q1, c1) = programBuilder.measure(q1);

  programBuilder.sink(q0);
  programBuilder.sink(q1);
  module = programBuilder.finalize({c0, c1});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();

  Value cr0;
  Value cr1;

  r0 = referenceBuilder.h(r0);
  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  r0 = referenceBuilder.reset(r0);
  r0 = referenceBuilder.h(r0);
  std::tie(r0, cr1) = referenceBuilder.measure(r0);
  referenceBuilder.sink(r0);

  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runReplaceClassicalControlsPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}
