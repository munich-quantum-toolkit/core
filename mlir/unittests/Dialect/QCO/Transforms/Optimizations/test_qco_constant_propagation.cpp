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
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

#include <numbers>

namespace {

using namespace mlir;
using namespace mlir::qco;

class QCOConstantPropagationTest : public testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder programBuilder;
  QCOProgramBuilder referenceBuilder;
  OwningOpRef<ModuleOp> module;
  OwningOpRef<ModuleOp> reference;

  QCOConstantPropagationTest()
      : programBuilder(&context), referenceBuilder(&context) {}

  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    programBuilder.initialize();
    referenceBuilder.initialize();
  }

  /**
   * @brief Adds the hadamardLiftingPass to the current context and runs it.
   */
  static LogicalResult runConstantPropagationPass(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(createConstantPropagation());
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

/**
 * @brief Test: This test checks if CNOTs or the controls of CNOTs are removed
 * if we can classically determine the ctrls value.
 */
TEST_F(QCOConstantPropagationTest, reducePosCtrls) {
  const auto iAttr = programBuilder.getF64FloatAttr(-0.3926991);
  Value i0 =
      arith::ConstantOp::create(programBuilder, programBuilder.getLoc(), iAttr);
  auto q = programBuilder.allocQubitRegister(4);
  q[0] = programBuilder.h(q[0]);
  q[0] = programBuilder.x(q[0]);
  q[0] = programBuilder.h(q[0]);
  programBuilder.cx(q[0], q[1]);
  q[2] = programBuilder.h(q[2]);
  q[2] = programBuilder.z(q[2]);
  q[2] = programBuilder.h(q[2]);
  auto [q2, q3] = programBuilder.crx(i0, q[2], q[3]);
  programBuilder.cry(0.3, q2, q3);
  module = programBuilder.finalize();

  const auto iAttrRef = referenceBuilder.getF64FloatAttr(-0.3926991);
  Value i0Ref = arith::ConstantOp::create(referenceBuilder,
                                          referenceBuilder.getLoc(), iAttrRef);
  auto qRef = referenceBuilder.allocQubitRegister(4);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[0] = referenceBuilder.x(qRef[0]);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[2] = referenceBuilder.h(qRef[2]);
  qRef[2] = referenceBuilder.z(qRef[2]);
  qRef[2] = referenceBuilder.h(qRef[2]);
  auto [q2Ref, q3Ref] = referenceBuilder.crx(i0Ref, qRef[2], qRef[3]);
  referenceBuilder.cry(0.3, q2Ref, q3Ref);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}