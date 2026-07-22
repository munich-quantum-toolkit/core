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
  static LogicalResult runQubitReusePass(ModuleOp module,
                                         bool liftMeasurements = false) {
    PassManager pm(module.getContext());
    if (liftMeasurements) {
      pm.addPass(createMeasurementLifting());
      pm.addPass(createReplaceClassicalControls());
    }
    pm.addPass(createReuseQubits());
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

// ==========================================================================
// Test qubit reuse only.
// ==========================================================================

/**
 * @brief A simple case where qubit reuse can be applied directly to go from 2
 * to 1 qubit.
 */
TEST_F(QCOQubitReuseTest, simpleReuse) {
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

  ASSERT_TRUE(runQubitReusePass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief A simple case where qubit reuse cannot be applied.
 */
TEST_F(QCOQubitReuseTest, noReuse) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();

  Value c0;
  Value c1;

  q0 = programBuilder.h(q0);
  std::tie(q0, q1) = programBuilder.cx(q0, q1);

  std::tie(q0, c0) = programBuilder.measure(q0);
  std::tie(q1, c1) = programBuilder.measure(q1);

  programBuilder.sink(q0);
  programBuilder.sink(q1);
  module = programBuilder.finalize({c0, c1});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();

  Value cr0;
  Value cr1;

  r0 = referenceBuilder.h(r0);
  std::tie(r0, r1) = referenceBuilder.cx(r0, r1);

  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  std::tie(r1, cr1) = referenceBuilder.measure(r1);

  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);

  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runQubitReusePass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test that partial qubit reuse is applied correctly in a context with
 * three qubits.
 */
TEST_F(QCOQubitReuseTest, reuseOneOfThreeQubits) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();

  Value c0;
  Value c1;
  Value c2;

  q0 = programBuilder.h(q0);
  std::tie(q0, q1) = programBuilder.cx(q0, q1);
  q2 = programBuilder.h(q2);

  std::tie(q0, c0) = programBuilder.measure(q0);
  std::tie(q1, c1) = programBuilder.measure(q1);
  std::tie(q2, c2) = programBuilder.measure(q2);

  programBuilder.sink(q0);
  programBuilder.sink(q1);
  programBuilder.sink(q2);
  module = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();

  Value cr0;
  Value cr1;
  Value cr2;

  r0 = referenceBuilder.h(r0);
  std::tie(r0, r1) = referenceBuilder.cx(r0, r1);

  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  std::tie(r1, cr1) = referenceBuilder.measure(r1);

  r1 = referenceBuilder.reset(r1);
  r1 = referenceBuilder.h(r1);

  std::tie(r1, cr2) = referenceBuilder.measure(r1);

  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);

  reference = referenceBuilder.finalize({cr0, cr1, cr2});

  ASSERT_TRUE(runQubitReusePass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test that qubit reuse is applied correctly in a context with three
 * qubits that can all be reused.
 */
TEST_F(QCOQubitReuseTest, reuseAllThreeQubits) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();

  Value c0;
  Value c1;
  Value c2;

  q0 = programBuilder.h(q0);
  q1 = programBuilder.h(q1);
  q2 = programBuilder.h(q2);

  std::tie(q0, c0) = programBuilder.measure(q0);
  std::tie(q1, c1) = programBuilder.measure(q1);
  std::tie(q2, c2) = programBuilder.measure(q2);

  programBuilder.sink(q0);
  programBuilder.sink(q1);
  programBuilder.sink(q2);
  module = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();

  Value cr0;
  Value cr1;
  Value cr2;

  r0 = referenceBuilder.h(r0);
  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  r0 = referenceBuilder.reset(r0);
  r0 = referenceBuilder.h(r0);
  std::tie(r0, cr1) = referenceBuilder.measure(r0);
  r0 = referenceBuilder.reset(r0);
  r0 = referenceBuilder.h(r0);
  std::tie(r0, cr2) = referenceBuilder.measure(r0);
  referenceBuilder.sink(r0);

  reference = referenceBuilder.finalize({cr0, cr1, cr2});

  ASSERT_TRUE(runQubitReusePass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test that qubit reuse can be applied even if the qubits are indirectly
 * connected.
 */
TEST_F(QCOQubitReuseTest, reuseIfPathExists) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();

  Value c0;
  Value c1;
  Value c2;

  q0 = programBuilder.h(q0);
  std::tie(q0, q1) = programBuilder.cx(q0, q1);
  std::tie(q0, q2) = programBuilder.cx(q0, q2);

  std::tie(q0, c0) = programBuilder.measure(q0);
  std::tie(q1, c1) = programBuilder.measure(q1);
  std::tie(q2, c2) = programBuilder.measure(q2);

  programBuilder.sink(q0);
  programBuilder.sink(q1);
  programBuilder.sink(q2);
  module = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();

  Value cr0;
  Value cr1;
  Value cr2;

  r0 = referenceBuilder.h(r0);
  std::tie(r0, r1) = referenceBuilder.cx(r0, r1);
  std::tie(r1, cr1) = referenceBuilder.measure(r1);
  r1 = referenceBuilder.reset(r1);
  std::tie(r0, r1) = referenceBuilder.cx(r0, r1);
  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  std::tie(r1, cr2) = referenceBuilder.measure(r1);
  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);

  reference = referenceBuilder.finalize({cr0, cr1, cr2});

  ASSERT_TRUE(runQubitReusePass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

// ==========================================================================
// Test qubit reuse with measurement lifting and control replacement.
// ==========================================================================

/**
 * @brief Test that qubit reuse can be applied after measurement lifting.
 */
TEST_F(QCOQubitReuseTest, singleReuseWithLift) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();

  Value c0;
  Value c1;

  q0 = programBuilder.x(q0);
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
  auto trueConstant = referenceBuilder.boolConstant(true);

  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  auto xorOp = arith::XOrIOp::create(
      referenceBuilder, referenceBuilder.getLoc(), cr0, trueConstant);
  cr0 = xorOp.getResult();
  r0 = referenceBuilder.reset(r0);
  r0 = referenceBuilder.h(r0);
  std::tie(r0, cr1) = referenceBuilder.measure(r0);
  referenceBuilder.sink(r0);

  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runQubitReusePass(module.get(), true).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test that qubit reuse can be applied after lifting measurements and
 * replacing controls.
 */
TEST_F(QCOQubitReuseTest, singleReuseWithControlLift) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();

  Value c0;
  Value c1;
  Value c2;

  q0 = programBuilder.h(q0);
  std::tie(q0, q1) = programBuilder.cx(q0, q1);
  std::tie(q2, q0) = programBuilder.cx(q2, q0);
  q0 = programBuilder.h(q0);

  std::tie(q0, c0) = programBuilder.measure(q0);
  std::tie(q1, c1) = programBuilder.measure(q1);
  std::tie(q2, c2) = programBuilder.measure(q2);

  programBuilder.sink(q0);
  programBuilder.sink(q1);
  programBuilder.sink(q2);
  module = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();

  Value cr0;
  Value cr1;
  Value cr2;

  r0 = referenceBuilder.h(r0);
  std::tie(r0, r1) = referenceBuilder.cx(r0, r1);
  std::tie(r1, cr1) = referenceBuilder.measure(r1);
  r1 = referenceBuilder.reset(r1);

  std::tie(r1, cr2) = referenceBuilder.measure(r1);

  r0 = referenceBuilder.qcoIf(
      cr2, r0, [&](Value qubit) { return referenceBuilder.x(qubit); });
  r0 = referenceBuilder.h(r0);

  std::tie(r0, cr0) = referenceBuilder.measure(r0);

  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);

  reference = referenceBuilder.finalize({cr0, cr1, cr2});

  ASSERT_TRUE(runQubitReusePass(module.get(), true).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test that qubit reuse can be applied with the help of measurement
 * lifting.
 */
TEST_F(QCOQubitReuseTest, singleReuseThroughLift) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();

  Value c0;
  Value c1;

  q0 = programBuilder.h(q0);
  std::tie(q0, q1) = programBuilder.cx(q0, q1);

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
  r0 = referenceBuilder.qcoIf(
      cr0, r0, [&](Value qubit) { return referenceBuilder.x(qubit); });
  std::tie(r0, cr1) = referenceBuilder.measure(r0);
  referenceBuilder.sink(r0);

  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runQubitReusePass(module.get(), true).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test that qubit reuse can be applied with the help of multi-step
 * measurement lifting.
 */
TEST_F(QCOQubitReuseTest, singleReuseThroughComplexLift) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();

  Value c0;
  Value c1;

  q0 = programBuilder.h(q0);
  std::tie(q0, q1) = programBuilder.cx(q0, q1);
  q0 = programBuilder.x(q0);

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
  auto trueConstant = referenceBuilder.boolConstant(true);

  r0 = referenceBuilder.h(r0);
  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  auto xorOp = arith::XOrIOp::create(
      referenceBuilder, referenceBuilder.getLoc(), cr0, trueConstant);

  r0 = referenceBuilder.reset(r0);
  r0 = referenceBuilder.qcoIf(
      cr0, r0, [&](Value qubit) { return referenceBuilder.x(qubit); });
  std::tie(r0, cr1) = referenceBuilder.measure(r0);
  referenceBuilder.sink(r0);

  cr0 = xorOp.getResult();
  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runQubitReusePass(module.get(), true).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test that qubit reuse can be applied after lifting measurements over
 * controlled gates if the qubit for which reuse should be applied was pulled
 * into an if/else block.
 */
TEST_F(QCOQubitReuseTest, multiReuseLiftOutOfIf) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  // As qubit reuse is built on a heuristic, the order of qubit allocations
  // matters. For this test, we need q0 to be allocated last.
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();
  auto q0 = programBuilder.allocQubit();

  Value c0;
  Value c1;
  Value c2;

  q0 = programBuilder.h(q0);
  std::tie(q1, q0) = programBuilder.cx(q1, q0);
  std::tie(q0, q2) = programBuilder.cx(q0, q2);

  std::tie(q0, c0) = programBuilder.measure(q0);
  std::tie(q1, c1) = programBuilder.measure(q1);
  std::tie(q2, c2) = programBuilder.measure(q2);

  programBuilder.sink(q0);
  programBuilder.sink(q1);
  programBuilder.sink(q2);
  module = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();

  Value cr0;
  Value cr1;
  Value cr2;

  std::tie(r0, cr1) = referenceBuilder.measure(r0);
  r0 = referenceBuilder.reset(r0);
  r0 = referenceBuilder.h(r0);
  r0 = referenceBuilder.qcoIf(
      cr1, r0, [&](Value qubit) { return referenceBuilder.x(qubit); });
  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  r0 = referenceBuilder.reset(r0);
  r0 = referenceBuilder.qcoIf(
      cr0, r0, [&](Value qubit) { return referenceBuilder.x(qubit); });
  std::tie(r0, cr2) = referenceBuilder.measure(r0);

  referenceBuilder.sink(r0);

  reference = referenceBuilder.finalize({cr0, cr1, cr2});

  ASSERT_TRUE(runQubitReusePass(module.get(), true).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}
