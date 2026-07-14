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

#include <numbers>
#include <tuple>

namespace {

using namespace mlir;
using namespace mlir::qco;

class QCOMeasurementLiftingTest : public testing::Test {

protected:
  MLIRContext context;
  QCOProgramBuilder programBuilder;
  QCOProgramBuilder referenceBuilder;
  OwningOpRef<ModuleOp> module;
  OwningOpRef<ModuleOp> reference;

  QCOMeasurementLiftingTest()
      : programBuilder(&context), referenceBuilder(&context) {}

  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  /**
   * @brief Adds the measurementLiftingPass to the current context and runs it.
   */
  static LogicalResult runMeasurementLiftingPass(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(createMeasurementLifting());
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

/**
 * @brief Test: Measurements on control bits can be lifted over the controlled
 * gates.
 */
TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverPositiveControl) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();

  std::tie(q1, q0) = programBuilder.cx(q1, q0);
  std::tie(q0, q1) = programBuilder.ch(q0, q1);
  std::tie(q0, q1) = programBuilder.cx(q0, q1);

  Value c0;
  Value c1;
  std::tie(q0, c0) = programBuilder.measure(q0);
  std::tie(q1, c1) = programBuilder.measure(q1);

  programBuilder.sink(q0);
  programBuilder.sink(q1);
  module = programBuilder.finalize({c0, c1});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();

  std::tie(r1, r0) = referenceBuilder.cx(r1, r0);
  Value cr0;
  Value cr1;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  std::tie(r0, r1) = referenceBuilder.ch(r0, r1);
  std::tie(r0, r1) = referenceBuilder.cx(r0, r1);

  std::tie(r1, cr1) = referenceBuilder.measure(r1);

  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);
  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Tests that lifting also works if there are multiple controls in
 * a controlled gate.
 */
TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverOneOfMultipleControls) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();

  SmallVector<Value> q12;
  SmallVector<Value> q0Vec;
  std::tie(q12, q0Vec) =
      programBuilder.ctrl({q1, q2}, {q0}, [&](ValueRange target) {
        return SmallVector{programBuilder.x(target[0])};
      });
  std::tie(q12, q0Vec) =
      programBuilder.ctrl({q12[1], q12[0]}, q0Vec, [&](ValueRange target) {
        return SmallVector{programBuilder.h(target[0])};
      });
  std::tie(q12, q0Vec) =
      programBuilder.ctrl({q12[1], q12[0]}, q0Vec, [&](ValueRange target) {
        return SmallVector{programBuilder.x(target[0])};
      });

  Value c1;
  std::tie(q1, c1) = programBuilder.measure(q12[0]);

  q0 = programBuilder.h(q0Vec[0]);
  q2 = programBuilder.h(q12[1]);

  Value c0;
  Value c2;
  std::tie(q0, c0) = programBuilder.measure(q0);
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
  auto r2 = referenceBuilder.allocQubit();

  Value cr1;
  std::tie(r1, cr1) = referenceBuilder.measure(r1);

  SmallVector<Value> r12;
  SmallVector<Value> r0Vec;
  std::tie(r12, r0Vec) =
      referenceBuilder.ctrl({r1, r2}, {r0}, [&](ValueRange target) {
        return SmallVector{referenceBuilder.x(target[0])};
      });
  std::tie(r12, r0Vec) =
      referenceBuilder.ctrl({r12[1], r12[0]}, r0Vec, [&](ValueRange target) {
        return SmallVector{referenceBuilder.h(target[0])};
      });
  std::tie(r12, r0Vec) =
      referenceBuilder.ctrl({r12[1], r12[0]}, r0Vec, [&](ValueRange target) {
        return SmallVector{referenceBuilder.x(target[0])};
      });

  r0 = referenceBuilder.h(r0Vec[0]);
  r2 = referenceBuilder.h(r12[1]);

  Value cr0;
  Value cr2;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  std::tie(r2, cr2) = referenceBuilder.measure(r2);

  referenceBuilder.sink(r0);
  referenceBuilder.sink(r12[0]);
  referenceBuilder.sink(r2);

  reference = referenceBuilder.finalize({cr0, cr1, cr2});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Tests that multiple measurements that each target a control
 * qubit of a controlled gate can be lifted over the controlled gate.
 */
TEST_F(QCOMeasurementLiftingTest,
       liftMeasurementMultipleOverOneControlledGate) {

  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();

  SmallVector<Value> q12;
  SmallVector<Value> q0Vec;
  std::tie(q12, q0Vec) =
      programBuilder.ctrl({q1, q2}, {q0}, [&](ValueRange target) {
        return SmallVector{programBuilder.x(target[0])};
      });

  Value c1;
  Value c2;
  std::tie(q1, c1) = programBuilder.measure(q12[0]);
  std::tie(q2, c2) = programBuilder.measure(q12[1]);

  programBuilder.sink(q0Vec[0]);
  programBuilder.sink(q1);
  programBuilder.sink(q2);
  module = programBuilder.finalize({c1, c2});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();
  auto r2 = referenceBuilder.allocQubit();

  Value cr1;
  Value cr2;
  std::tie(r1, cr1) = referenceBuilder.measure(r1);
  std::tie(r2, cr2) = referenceBuilder.measure(r2);

  SmallVector<Value> r12;
  SmallVector<Value> r0Vec;
  std::tie(r12, r0Vec) =
      referenceBuilder.ctrl({r1, r2}, {r0}, [&](ValueRange target) {
        return SmallVector{referenceBuilder.x(target[0])};
      });

  referenceBuilder.sink(r0Vec[0]);
  referenceBuilder.sink(r12[0]);
  referenceBuilder.sink(r12[1]);
  reference = referenceBuilder.finalize({cr1, cr2});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Tests that a measurement can also be lifted over the control of
 * a parametrized gate.
 */
TEST_F(QCOMeasurementLiftingTest,
       liftMeasurementOverControlledParametrizedGate) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();

  std::tie(q0, q1) = programBuilder.crx(std::numbers::pi / 2, q0, q1);

  Value c0;
  Value c1;
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
  std::tie(r0, cr0) = referenceBuilder.measure(r0);

  std::tie(r0, r1) = referenceBuilder.crx(std::numbers::pi / 2, r0, r1);

  std::tie(r1, cr1) = referenceBuilder.measure(r1);

  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);
  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Tests lifting a measurement over a single X (anti-diagonal)
 * gate.
 */
TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverSingleX) {

  programBuilder.initialize({programBuilder.getI1Type()});
  auto q = programBuilder.allocQubit();
  q = programBuilder.x(q);
  Value c;
  std::tie(q, c) = programBuilder.measure(q);
  q = programBuilder.h(q);
  programBuilder.sink(q);
  module = programBuilder.finalize(c);

  referenceBuilder.initialize({referenceBuilder.getI1Type()});
  auto r = referenceBuilder.allocQubit();
  auto trueConstant = referenceBuilder.boolConstant(true);
  Value cr;
  std::tie(r, cr) = referenceBuilder.measure(r);
  r = referenceBuilder.h(r);

  auto xorOp = arith::XOrIOp::create(
      referenceBuilder, referenceBuilder.getLoc(), cr, trueConstant);
  referenceBuilder.sink(r);
  reference = referenceBuilder.finalize(xorOp.getResult());

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Tests lifting a measurement over a single Y (anti-diagonal)
 * gate.
 */
TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverSingleY) {
  programBuilder.initialize({programBuilder.getI1Type()});
  auto q = programBuilder.allocQubit();
  q = programBuilder.y(q);
  Value c;
  std::tie(q, c) = programBuilder.measure(q);
  programBuilder.sink(q);
  module = programBuilder.finalize({c});

  referenceBuilder.initialize({referenceBuilder.getI1Type()});
  auto r = referenceBuilder.allocQubit();
  auto trueConstant = referenceBuilder.boolConstant(true);
  Value cr;
  std::tie(r, cr) = referenceBuilder.measure(r);
  auto xorOp = arith::XOrIOp::create(
      referenceBuilder, referenceBuilder.getLoc(), cr, trueConstant);
  referenceBuilder.sink(r);
  reference = referenceBuilder.finalize({xorOp.getResult()});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Tests lifting a measurement over different diagonal phase-gates.
 */
TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverPhaseGates) {
  programBuilder.initialize({programBuilder.getI1Type()});
  auto q = programBuilder.allocQubit();
  q = programBuilder.id(q);
  q = programBuilder.z(q);
  q = programBuilder.s(q);
  q = programBuilder.sdg(q);
  q = programBuilder.t(q);
  q = programBuilder.tdg(q);
  q = programBuilder.p(std::numbers::pi / 2, q);
  q = programBuilder.rz(std::numbers::pi / 2, q);
  Value c;
  std::tie(q, c) = programBuilder.measure(q);
  programBuilder.sink(q);
  module = programBuilder.finalize({c});

  referenceBuilder.initialize({referenceBuilder.getI1Type()});
  auto r = referenceBuilder.allocQubit();
  Value cr;
  std::tie(r, cr) = referenceBuilder.measure(r);
  referenceBuilder.sink(r);
  reference = referenceBuilder.finalize({cr});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Tests lifting a measurement over multiple anti-diagonal gates.
 */
TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverMultipleXY) {
  programBuilder.initialize({programBuilder.getI1Type()});
  auto q = programBuilder.allocQubit();
  q = programBuilder.x(q);
  q = programBuilder.y(q);
  Value c;
  std::tie(q, c) = programBuilder.measure(q);
  programBuilder.sink(q);
  module = programBuilder.finalize({c});

  referenceBuilder.initialize({referenceBuilder.getI1Type()});
  auto r = referenceBuilder.allocQubit();
  Value cr;
  std::tie(r, cr) = referenceBuilder.measure(r);
  referenceBuilder.sink(r);
  reference = referenceBuilder.finalize({cr});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Tests lifting a measurement over multiple anti-diagonal and
 * controlled gates.
 */
TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverXAndControlledGates) {
  programBuilder.initialize({programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();

  std::tie(q0, q1) = programBuilder.cy(q0, q1);
  q0 = programBuilder.x(q0);
  std::tie(q0, q1) = programBuilder.cy(q0, q1);
  q0 = programBuilder.x(q0);

  Value c0;
  std::tie(q0, c0) = programBuilder.measure(q0);

  programBuilder.sink(q0);
  programBuilder.sink(q1);
  module = programBuilder.finalize({c0});

  referenceBuilder.initialize({referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();

  Value cr0;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);

  std::tie(r0, r1) = referenceBuilder.cy(r0, r1);
  r0 = referenceBuilder.x(r0);
  std::tie(r0, r1) = referenceBuilder.cy(r0, r1);

  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);
  reference = referenceBuilder.finalize({cr0});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Tests lifting a measurement over a controlled diagonal gate.
 */
TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverDiagonalGateInControl) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();

  std::tie(q0, q1) = programBuilder.cz(q0, q1);

  Value c0;
  Value c1;
  std::tie(q0, c0) = programBuilder.measure(q0);
  std::tie(q1, c1) = programBuilder.measure(q1);

  programBuilder.sink(q0);
  programBuilder.sink(q1);
  module = programBuilder.finalize({c0, c1});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), programBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();

  Value cr0;
  Value cr1;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  std::tie(r1, cr1) = referenceBuilder.measure(r1);

  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);
  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Tests that a measurement is not lifted over a controlled
 * sequence gate if there are multiple gates inside the control block.
 */
TEST_F(QCOMeasurementLiftingTest, dontLiftMeasurementMultipleGatesInControl) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto q0S0 = programBuilder.allocQubit();
  auto q1S0 = programBuilder.allocQubit();

  auto [q0S1, q1S1] = programBuilder.ctrl({q0S0}, q1S0, [&](Value target) {
    auto t = programBuilder.z(target);
    return programBuilder.x(t);
  });

  auto [q0S2, c0] = programBuilder.measure(q0S1);
  auto [q1S2, c1] = programBuilder.measure(q1S1);

  programBuilder.sink(q0S2);
  programBuilder.sink(q1S2);
  module = programBuilder.finalize({c0, c1});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0S0 = referenceBuilder.allocQubit();
  auto r1S0 = referenceBuilder.allocQubit();

  auto [r0S1, cr0] = referenceBuilder.measure(r0S0);

  auto [r0S2, r1S1] = referenceBuilder.ctrl({r0S1}, r1S0, [&](Value target) {
    auto t = referenceBuilder.z(target);
    return referenceBuilder.x(t);
  });

  auto [r1S2, cr1] = referenceBuilder.measure(r1S1);

  referenceBuilder.sink(r0S2);
  referenceBuilder.sink(r1S2);
  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Tests lifting a measurement over an inverted phase gate.
 */
TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverInvertedPhaseGates) {
  programBuilder.initialize({programBuilder.getI1Type()});
  auto q = programBuilder.allocQubit();

  q = programBuilder.inv(
      q, [&](Value target) { return programBuilder.s(target); });

  Value c;
  std::tie(q, c) = programBuilder.measure(q);
  programBuilder.sink(q);
  module = programBuilder.finalize({c});

  referenceBuilder.initialize({referenceBuilder.getI1Type()});
  auto r = referenceBuilder.allocQubit();
  Value cr;
  std::tie(r, cr) = referenceBuilder.measure(r);
  referenceBuilder.sink(r);
  reference = referenceBuilder.finalize({cr});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}
