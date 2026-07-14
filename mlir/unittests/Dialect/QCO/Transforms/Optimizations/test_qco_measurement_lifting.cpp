/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

//
// Created by damian on 5/21/26.
//

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

TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverPositiveControl) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0S0 = programBuilder.allocQubit();
  auto q1S0 = programBuilder.allocQubit();

  auto [q1S1, q0S1] = programBuilder.cx(q1S0, q0S0);
  auto [q0S2, q1S2] = programBuilder.ch(q0S1, q1S1);
  auto [q0S3, q1S3] = programBuilder.cx(q0S2, q1S2);

  auto [q0S4, c0] = programBuilder.measure(q0S3);
  auto [q1S4, c1] = programBuilder.measure(q1S3);

  programBuilder.sink(q0S4);
  programBuilder.sink(q1S4);
  module = programBuilder.finalize({c0, c1});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0S0 = referenceBuilder.allocQubit();
  auto r1S0 = referenceBuilder.allocQubit();

  auto [r1S1, r0S1] = referenceBuilder.cx(r1S0, r0S0);
  auto [r0S2, cr0] = referenceBuilder.measure(r0S1);
  auto [r0S3, r1S2] = referenceBuilder.ch(r0S2, r1S1);
  auto [r0S4, r1S3] = referenceBuilder.cx(r0S3, r1S2);

  auto [r1S4, cr1] = referenceBuilder.measure(r1S3);

  referenceBuilder.sink(r0S4);
  referenceBuilder.sink(r1S4);
  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverOneOfMultipleControls) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto q0S0 = programBuilder.allocQubit();
  auto q1S0 = programBuilder.allocQubit();
  auto q2S0 = programBuilder.allocQubit();

  auto [q12_0, q0S1] =
      programBuilder.ctrl({q1S0, q2S0}, {q0S0}, [&](const ValueRange target) {
        return SmallVector{programBuilder.x(target[0])};
      });
  auto [q12_1, q0S2] = programBuilder.ctrl(
      {q12_0[1], q12_0[0]}, q0S1, [&](const ValueRange target) {
        return SmallVector{programBuilder.h(target[0])};
      });
  auto [q12_2, q0S3] = programBuilder.ctrl(
      {q12_1[1], q12_1[0]}, q0S2, [&](const ValueRange target) {
        return SmallVector{programBuilder.x(target[0])};
      });

  auto [q1S4, c1] = programBuilder.measure(q12_2[0]);

  auto q0S4 = programBuilder.h(q0S3[0]);
  auto q2S4 = programBuilder.h(q12_2[1]);

  auto [q0S5, c0] = programBuilder.measure(q0S4);
  auto [q2S5, c2] = programBuilder.measure(q2S4);

  programBuilder.sink(q0S5);
  programBuilder.sink(q1S4);
  programBuilder.sink(q2S5);

  module = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r0S0 = referenceBuilder.allocQubit();
  auto r1S0 = referenceBuilder.allocQubit();
  auto r2S0 = referenceBuilder.allocQubit();

  auto [r1S1, cr1] = referenceBuilder.measure(r1S0);

  auto [r12_0, r0S1] =
      referenceBuilder.ctrl({r1S1, r2S0}, {r0S0}, [&](const ValueRange target) {
        return SmallVector{referenceBuilder.x(target[0])};
      });
  auto [r12_1, r0S2] = referenceBuilder.ctrl(
      {r12_0[1], r12_0[0]}, r0S1, [&](const ValueRange target) {
        return SmallVector{referenceBuilder.h(target[0])};
      });
  auto [r12_2, r0S3] = referenceBuilder.ctrl(
      {r12_1[1], r12_1[0]}, r0S2, [&](const ValueRange target) {
        return SmallVector{referenceBuilder.x(target[0])};
      });

  auto r0S4 = referenceBuilder.h(r0S3[0]);
  auto r2S4 = referenceBuilder.h(r12_2[1]);

  auto [r0S5, cr0] = referenceBuilder.measure(r0S4);
  auto [r2S5, cr2] = referenceBuilder.measure(r2S4);

  referenceBuilder.sink(r0S5);
  referenceBuilder.sink(r12_2[0]);
  referenceBuilder.sink(r2S5);

  reference = referenceBuilder.finalize({cr0, cr1, cr2});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest,
       liftMeasurementMultipleOverOneControlledGate) {

  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0S0 = programBuilder.allocQubit();
  auto q1S0 = programBuilder.allocQubit();
  auto q2S0 = programBuilder.allocQubit();

  auto [q12_0, q0S1] =
      programBuilder.ctrl({q1S0, q2S0}, {q0S0}, [&](const ValueRange target) {
        return SmallVector{programBuilder.x(target[0])};
      });

  auto [q1S1, c1] = programBuilder.measure(q12_0[0]);
  auto [q2S1, c2] = programBuilder.measure(q12_0[1]);

  programBuilder.sink(q0S1[0]);
  programBuilder.sink(q1S1);
  programBuilder.sink(q2S1);
  module = programBuilder.finalize({c1, c2});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0S0 = referenceBuilder.allocQubit();
  auto r1S0 = referenceBuilder.allocQubit();
  auto r2S0 = referenceBuilder.allocQubit();

  auto [r1S1, cr1] = referenceBuilder.measure(r1S0);
  auto [r2S1, cr2] = referenceBuilder.measure(r2S0);

  auto [r12_0, r0S1] =
      referenceBuilder.ctrl({r1S1, r2S1}, {r0S0}, [&](const ValueRange target) {
        return SmallVector{referenceBuilder.x(target[0])};
      });

  referenceBuilder.sink(r0S1[0]);
  referenceBuilder.sink(r12_0[0]);
  referenceBuilder.sink(r12_0[1]);
  reference = referenceBuilder.finalize({cr1, cr2});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest,
       liftMeasurementOverControlledParametrizedGate) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0S0 = programBuilder.allocQubit();
  auto q1S0 = programBuilder.allocQubit();

  auto [q0S1, q1S1] = programBuilder.crx(std::numbers::pi / 2, q0S0, q1S0);

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

  auto [r0S2, r1S1] = referenceBuilder.crx(std::numbers::pi / 2, r0S1, r1S0);

  auto [r1S2, cr1] = referenceBuilder.measure(r1S1);

  referenceBuilder.sink(r0S2);
  referenceBuilder.sink(r1S2);
  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverSingleX) {

  programBuilder.initialize({programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.x(q0);
  auto [q2, c] = programBuilder.measure(q1);
  auto q3 = programBuilder.h(q2);
  programBuilder.sink(q3);
  module = programBuilder.finalize(c);

  referenceBuilder.initialize({referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto trueConstant = referenceBuilder.boolConstant(true);
  auto [r1, cr] = referenceBuilder.measure(r0);
  auto r2 = referenceBuilder.h(r1);

  auto xorOp = arith::XOrIOp::create(
      referenceBuilder, referenceBuilder.getLoc(), cr, trueConstant);
  referenceBuilder.sink(r2);
  reference = referenceBuilder.finalize(xorOp.getResult());

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverSingleY) {
  programBuilder.initialize({programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.y(q0);
  auto [q2, c] = programBuilder.measure(q1);
  programBuilder.sink(q2);
  module = programBuilder.finalize({c});

  referenceBuilder.initialize({referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto trueConstant = referenceBuilder.boolConstant(true);
  auto [r1, cr] = referenceBuilder.measure(r0);
  auto xorOp = arith::XOrIOp::create(
      referenceBuilder, referenceBuilder.getLoc(), cr, trueConstant);
  referenceBuilder.sink(r1);
  reference = referenceBuilder.finalize({xorOp.getResult()});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverPhaseGates) {
  programBuilder.initialize({programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.id(q0);
  auto q2 = programBuilder.z(q1);
  auto q3 = programBuilder.s(q2);
  auto q4 = programBuilder.sdg(q3);
  auto q5 = programBuilder.t(q4);
  auto q6 = programBuilder.tdg(q5);
  auto q7 = programBuilder.p(std::numbers::pi / 2, q6);
  auto q8 = programBuilder.rz(std::numbers::pi / 2, q7);
  auto [q9, c] = programBuilder.measure(q8);
  programBuilder.sink(q9);
  module = programBuilder.finalize({c});

  referenceBuilder.initialize({referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto [r1, cr] = referenceBuilder.measure(r0);
  referenceBuilder.sink(r1);
  reference = referenceBuilder.finalize({cr});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverMultipleXY) {
  programBuilder.initialize({programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.x(q0);
  auto q2 = programBuilder.y(q1);
  auto [q3, c] = programBuilder.measure(q2);
  programBuilder.sink(q3);
  module = programBuilder.finalize({c});

  referenceBuilder.initialize({referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto [r1, cr] = referenceBuilder.measure(r0);
  referenceBuilder.sink(r1);
  reference = referenceBuilder.finalize({cr});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverXAndControlledGates) {
  programBuilder.initialize({programBuilder.getI1Type()});
  auto q0S0 = programBuilder.allocQubit();
  auto q1S0 = programBuilder.allocQubit();

  auto [q0S1, q1S1] = programBuilder.cy(q0S0, q1S0);
  auto q0S2 = programBuilder.x(q0S1);
  auto [q0S3, q1S2] = programBuilder.cy(q0S2, q1S1);
  auto q0S4 = programBuilder.x(q0S3);

  auto [q0S5, c0] = programBuilder.measure(q0S4);

  programBuilder.sink(q0S5);
  programBuilder.sink(q1S2);
  module = programBuilder.finalize({c0});

  referenceBuilder.initialize({referenceBuilder.getI1Type()});
  auto r0S0 = referenceBuilder.allocQubit();
  auto r1S0 = referenceBuilder.allocQubit();

  auto [r0S1, cr0] = referenceBuilder.measure(r0S0);

  auto [r0S2, r1S1] = referenceBuilder.cx(r0S1, r1S0);
  auto r0S3 = referenceBuilder.x(r0S2);
  auto [r0S4, r1S2] = referenceBuilder.cx(r0S3, r1S1);

  referenceBuilder.sink(r0S4);
  referenceBuilder.sink(r1S2);
  reference = referenceBuilder.finalize({cr0});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverDiagonalGateInControl) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0S0 = programBuilder.allocQubit();
  auto q1S0 = programBuilder.allocQubit();

  auto [q0S1, q1S1] = programBuilder.cz(q0S0, q1S0);

  auto [q0S2, c0] = programBuilder.measure(q0S1);
  auto [q1S2, c1] = programBuilder.measure(q1S1);

  programBuilder.sink(q0S2);
  programBuilder.sink(q1S2);
  module = programBuilder.finalize({c0, c1});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), programBuilder.getI1Type()});
  auto r0S0 = referenceBuilder.allocQubit();
  auto r1S0 = referenceBuilder.allocQubit();

  auto [r0S1, cr0] = referenceBuilder.measure(r0S0);
  auto [r1S1, cr1] = referenceBuilder.measure(r1S0);

  referenceBuilder.sink(r0S1);
  referenceBuilder.sink(r1S1);
  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

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
