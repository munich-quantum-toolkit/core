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

    programBuilder.initialize();
    referenceBuilder.initialize();
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

  static Value i1ToI64(Value i1Value, ImplicitLocOpBuilder& builder) {
    return arith::ExtUIOp::create(builder, builder.getI64Type(), i1Value)
        .getResult();
  }
};

} // namespace

TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverPositiveControl) {
  auto q0_0 = programBuilder.allocQubit();
  auto q1_0 = programBuilder.allocQubit();

  auto [q1_1, q0_1] = programBuilder.cx(q1_0, q0_0);
  auto [q0_2, q1_2] = programBuilder.ch(q0_1, q1_1);
  auto [q0_3, q1_3] = programBuilder.cx(q0_2, q1_2);

  auto [q0_4, c0] = programBuilder.measure(q0_3);
  auto [q1_4, c1] = programBuilder.measure(q1_3);

  programBuilder.sink(q0_4);
  programBuilder.sink(q1_4);
  module = programBuilder.finalize();

  auto r0_0 = referenceBuilder.allocQubit();
  auto r1_0 = referenceBuilder.allocQubit();

  auto [r1_1, r0_1] = referenceBuilder.cx(r1_0, r0_0);
  auto [r0_2, cr0] = referenceBuilder.measure(r0_1);
  auto [r0_3, r1_2] = referenceBuilder.ch(r0_2, r1_1);
  auto [r0_4, r1_3] = referenceBuilder.cx(r0_3, r1_2);

  auto [r1_4, cr1] = referenceBuilder.measure(r1_3);

  referenceBuilder.sink(r0_4);
  referenceBuilder.sink(r1_4);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverOneOfMultipleControls) {
  auto q0_0 = programBuilder.allocQubit();
  auto q1_0 = programBuilder.allocQubit();
  auto q2_0 = programBuilder.allocQubit();

  auto [q12_0, q0_1] =
      programBuilder.ctrl({q1_0, q2_0}, {q0_0}, [&](const ValueRange target) {
        return SmallVector{programBuilder.x(target[0])};
      });
  auto [q12_1, q0_2] = programBuilder.ctrl(
      {q12_0[1], q12_0[0]}, q0_1, [&](const ValueRange target) {
        return SmallVector{programBuilder.h(target[0])};
      });
  auto [q12_2, q0_3] = programBuilder.ctrl(
      {q12_1[1], q12_1[0]}, q0_2, [&](const ValueRange target) {
        return SmallVector{programBuilder.x(target[0])};
      });

  auto [q1_4, c1] = programBuilder.measure(q12_2[0]);

  auto q0_4 = programBuilder.h(q0_3[0]);
  auto q2_4 = programBuilder.h(q12_2[1]);

  auto [q0_5, c0] = programBuilder.measure(q0_4);
  auto [q2_5, c2] = programBuilder.measure(q2_4);

  programBuilder.sink(q0_5);
  programBuilder.sink(q1_4);
  programBuilder.sink(q2_5);

  module = programBuilder.finalize();

  auto r0_0 = referenceBuilder.allocQubit();
  auto r1_0 = referenceBuilder.allocQubit();
  auto r2_0 = referenceBuilder.allocQubit();

  auto [r1_1, cr1] = referenceBuilder.measure(r1_0);

  auto [r12_0, r0_1] =
      referenceBuilder.ctrl({r1_1, r2_0}, {r0_0}, [&](const ValueRange target) {
        return SmallVector{referenceBuilder.x(target[0])};
      });
  auto [r12_1, r0_2] = referenceBuilder.ctrl(
      {r12_0[1], r12_0[0]}, r0_1, [&](const ValueRange target) {
        return SmallVector{referenceBuilder.h(target[0])};
      });
  auto [r12_2, r0_3] = referenceBuilder.ctrl(
      {r12_1[1], r12_1[0]}, r0_2, [&](const ValueRange target) {
        return SmallVector{referenceBuilder.x(target[0])};
      });

  auto r0_4 = referenceBuilder.h(r0_3[0]);
  auto r2_4 = referenceBuilder.h(r12_2[1]);

  auto [r0_5, cr0] = referenceBuilder.measure(r0_4);
  auto [r2_5, cr2] = referenceBuilder.measure(r2_4);

  referenceBuilder.sink(r0_5);
  referenceBuilder.sink(r12_2[0]);
  referenceBuilder.sink(r2_5);

  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest,
       liftMeasurementMultipleOverOneControlledGate) {
  auto q0_0 = programBuilder.allocQubit();
  auto q1_0 = programBuilder.allocQubit();
  auto q2_0 = programBuilder.allocQubit();

  auto [q12_0, q0_1] =
      programBuilder.ctrl({q1_0, q2_0}, {q0_0}, [&](const ValueRange target) {
        return SmallVector{programBuilder.x(target[0])};
      });

  auto [q1_1, c1] = programBuilder.measure(q12_0[0]);
  auto [q2_1, c2] = programBuilder.measure(q12_0[1]);

  programBuilder.sink(q0_1[0]);
  programBuilder.sink(q1_1);
  programBuilder.sink(q2_1);
  module = programBuilder.finalize();

  auto r0_0 = referenceBuilder.allocQubit();
  auto r1_0 = referenceBuilder.allocQubit();
  auto r2_0 = referenceBuilder.allocQubit();

  auto [r1_1, cr1] = programBuilder.measure(r1_0);
  auto [r2_1, cr2] = programBuilder.measure(r2_0);

  auto [r12_0, r0_1] =
      programBuilder.ctrl({r1_1, r2_1}, {r0_0}, [&](const ValueRange target) {
        return SmallVector{programBuilder.x(target[0])};
      });

  referenceBuilder.sink(r0_1[0]);
  referenceBuilder.sink(r12_0[0]);
  referenceBuilder.sink(r12_0[1]);
  module = referenceBuilder.finalize();

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest,
       liftMeasurementOverControlledParametrizedGate) {
  auto q0_0 = programBuilder.allocQubit();
  auto q1_0 = programBuilder.allocQubit();

  auto [q0_1, q1_1] = programBuilder.crx(std::numbers::pi / 2, q0_0, q1_0);

  auto [q0_2, c0] = programBuilder.measure(q0_1);
  auto [q1_2, c1] = programBuilder.measure(q1_1);

  programBuilder.sink(q0_2);
  programBuilder.sink(q1_2);
  module = programBuilder.finalize();

  auto r0_0 = referenceBuilder.allocQubit();
  auto r1_0 = referenceBuilder.allocQubit();

  auto [r0_1, cr0] = referenceBuilder.measure(r0_0);

  auto [r0_2, r1_1] = referenceBuilder.crx(std::numbers::pi / 2, r0_1, r1_0);

  auto [r1_2, cr1] = referenceBuilder.measure(r1_1);

  referenceBuilder.sink(r0_2);
  referenceBuilder.sink(r1_2);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverSingleX) {
  auto q_0 = programBuilder.allocQubit();
  auto q_1 = programBuilder.x(q_0);
  auto [q_2, c] = programBuilder.measure(q_1);
  programBuilder.sink(q_2);
  module = programBuilder.finalize(i1ToI64(c, programBuilder));

  auto r_0 = referenceBuilder.allocQubit();
  auto true_constant = referenceBuilder.boolConstant(true);
  auto [r_1, cr] = referenceBuilder.measure(r_0);

  auto xorOp = arith::XOrIOp::create(
      referenceBuilder, referenceBuilder.getLoc(), cr, true_constant);
  referenceBuilder.sink(r_1);
  reference =
      referenceBuilder.finalize(i1ToI64(xorOp.getResult(), referenceBuilder));

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  reference.get().dump();
  module.get().dump();

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverSingleY) {
  auto q_0 = programBuilder.allocQubit();
  auto q_1 = programBuilder.y(q_0);
  auto [q_2, c] = programBuilder.measure(q_1);
  programBuilder.sink(q_2);
  module = programBuilder.finalize();

  auto r_0 = referenceBuilder.allocQubit();
  auto true_constant = referenceBuilder.boolConstant(true);
  auto [r_1, cr] = referenceBuilder.measure(r_0);
  referenceBuilder.insert(arith::XOrIOp::create(
      referenceBuilder, referenceBuilder.getLoc(), cr, true_constant));
  referenceBuilder.sink(r_1);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverPhaseGates) {
  auto q_0 = programBuilder.allocQubit();
  auto q_1 = programBuilder.id(q_0);
  auto q_2 = programBuilder.z(q_1);
  auto q_3 = programBuilder.s(q_2);
  auto q_4 = programBuilder.sdg(q_3);
  auto q_5 = programBuilder.t(q_4);
  auto q_6 = programBuilder.tdg(q_5);
  auto q_7 = programBuilder.p(std::numbers::pi / 2, q_6);
  auto q_8 = programBuilder.rz(std::numbers::pi / 2, q_7);
  auto [q_9, c] = programBuilder.measure(q_8);
  programBuilder.sink(q_9);
  module = programBuilder.finalize();

  auto r_0 = referenceBuilder.allocQubit();
  auto [r_1, cr] = referenceBuilder.measure(r_0);
  referenceBuilder.sink(r_1);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverMultipleXY) {
  auto q_0 = programBuilder.allocQubit();
  auto q_1 = programBuilder.x(q_0);
  auto q_2 = programBuilder.y(q_1);
  auto [q_3, c] = programBuilder.measure(q_2);
  programBuilder.sink(q_3);
  module = programBuilder.finalize();

  auto r_0 = referenceBuilder.allocQubit();
  auto [r_1, cr] = referenceBuilder.measure(r_0);
  referenceBuilder.sink(r_1);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverXAndControlledGates) {
  auto q0_0 = programBuilder.allocQubit();
  auto q1_0 = programBuilder.allocQubit();

  auto [q0_1, q1_1] = programBuilder.cy(q0_0, q1_0);
  auto q0_2 = programBuilder.x(q0_1);
  auto [q0_3, q1_2] = programBuilder.cy(q0_2, q1_1);
  auto q0_4 = programBuilder.x(q0_3);

  auto [q0_5, c0] = programBuilder.measure(q0_4);

  programBuilder.sink(q0_5);
  programBuilder.sink(q1_2);
  module = programBuilder.finalize();

  auto r0_0 = referenceBuilder.allocQubit();
  auto r1_0 = referenceBuilder.allocQubit();

  auto [r0_1, cr0] = referenceBuilder.measure(r0_0);

  auto [r0_2, r1_1] = referenceBuilder.cx(r0_1, r1_0);
  auto r0_3 = referenceBuilder.x(r0_2);
  auto [r0_4, r1_2] = referenceBuilder.cx(r0_3, r1_1);

  referenceBuilder.sink(r0_4);
  referenceBuilder.sink(r1_2);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOMeasurementLiftingTest, liftMeasurementOverDiagonalGateInControl) {
  auto q0_0 = programBuilder.allocQubit();
  auto q1_0 = programBuilder.allocQubit();

  auto [q0_1, q1_1] = programBuilder.cz(q0_0, q1_0);

  auto [q0_2, c0] = programBuilder.measure(q0_1);

  programBuilder.sink(q0_2);
  programBuilder.sink(q1_1);
  module = programBuilder.finalize();

  auto r0_0 = referenceBuilder.allocQubit();
  auto r1_0 = referenceBuilder.allocQubit();

  auto [r0_1, cr0] = referenceBuilder.measure(r0_0);

  referenceBuilder.sink(r0_1);
  referenceBuilder.sink(r1_0);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runMeasurementLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}
