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

class QCOReplaceClassicalControlsTest : public testing::Test {

protected:
  MLIRContext context;
  QCOProgramBuilder programBuilder;
  QCOProgramBuilder referenceBuilder;
  OwningOpRef<ModuleOp> program;
  OwningOpRef<ModuleOp> reference;

  QCOReplaceClassicalControlsTest()
      : programBuilder(&context), referenceBuilder(&context) {}

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
  runReplaceClassicalControlsPass(ModuleOp program,
                                  bool liftMeasurements = false) {
    PassManager pm(program.getContext());
    pm.addPass(createReplaceClassicalControls());
    if (liftMeasurements) {
      pm.addPass(createMeasurementLifting());
    }
    pm.addPass(createCanonicalizerPass());
    return pm.run(program);
  }

  /**
   * @brief Adds the canonicalizerPass to the current context and runs it.
   */
  static LogicalResult runCanonicalizerPass(ModuleOp program) {
    PassManager pm(program.getContext());
    pm.addPass(createCanonicalizerPass());
    return pm.run(program);
  }
};

} // namespace

/**
 * @brief Test: Tests replacing a classically controlled gate where there is
 * only one control.
 */
TEST_F(QCOReplaceClassicalControlsTest, replaceClassicalControlsOnlyControl) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  q0 = programBuilder.h(q0);
  q1 = programBuilder.h(q1);

  Value c0;
  std::tie(q0, c0) = programBuilder.measure(q0);
  std::tie(q0, q1) = programBuilder.cx(q0, q1);
  Value c1;
  std::tie(q1, c1) = programBuilder.measure(q1);

  programBuilder.sink(q0);
  programBuilder.sink(q1);
  program = programBuilder.finalize({c0, c1});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();
  r0 = referenceBuilder.h(r0);
  r1 = referenceBuilder.h(r1);

  Value cr0;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);

  r1 = referenceBuilder.qcoIf(
      cr0, r1, [&](Value qubit) -> Value { return referenceBuilder.x(qubit); });
  Value cr1;
  std::tie(r1, cr1) = referenceBuilder.measure(r1);
  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);

  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runReplaceClassicalControlsPass(program.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/**
 * @brief Test: Tests replacing a classically controlled gate where only one of
 * two controls can be replaced.
 */
TEST_F(QCOReplaceClassicalControlsTest,
       replaceClassicalControlsOneOfTwoControls) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();
  q0 = programBuilder.h(q0);
  q1 = programBuilder.h(q1);
  q2 = programBuilder.h(q2);

  Value c0;
  std::tie(q0, c0) = programBuilder.measure(q0);

  SmallVector<Value> q01;
  SmallVector<Value> q2Vec;
  std::tie(q01, q2Vec) = programBuilder.ctrl(
      {q0, q1}, {q2}, [&](ValueRange targets) -> SmallVector<Value> {
        return SmallVector<Value>{programBuilder.x(targets[0])};
      });

  Value c1;
  std::tie(q1, c1) = programBuilder.measure(q01[1]);
  Value c2;
  std::tie(q2, c2) = programBuilder.measure(q2Vec[0]);

  programBuilder.sink(q01[0]);
  programBuilder.sink(q1);
  programBuilder.sink(q2);
  program = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();
  auto r2 = referenceBuilder.allocQubit();
  r0 = referenceBuilder.h(r0);
  r1 = referenceBuilder.h(r1);
  r2 = referenceBuilder.h(r2);

  Value cr0;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);

  SmallVector<Value> r12 = referenceBuilder.qcoIf(
      cr0, {r1, r2}, [&](ValueRange qubits) -> SmallVector<Value> {
        Value t1 = qubits[0];
        Value t2 = qubits[1];
        std::tie(t1, t2) = referenceBuilder.cx(t1, t2);
        return SmallVector<Value>{t1, t2};
      });
  Value cr1;
  std::tie(r1, cr1) = referenceBuilder.measure(r12[0]);
  Value cr2;
  std::tie(r2, cr2) = referenceBuilder.measure(r12[1]);
  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);
  referenceBuilder.sink(r2);

  reference = referenceBuilder.finalize({cr0, cr1, cr2});

  ASSERT_TRUE(runReplaceClassicalControlsPass(program.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/**
 * @brief Test: Tests replacing a classically controlled gate where both of the
 * two controls can be replaced.
 */
TEST_F(QCOReplaceClassicalControlsTest,
       replaceClassicalControlsTwoOfTwoControls) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();
  q0 = programBuilder.h(q0);
  q1 = programBuilder.h(q1);
  q2 = programBuilder.h(q2);

  Value c0;
  std::tie(q0, c0) = programBuilder.measure(q0);
  Value c1;
  std::tie(q1, c1) = programBuilder.measure(q1);

  SmallVector<Value> q01;
  SmallVector<Value> q2Vec;
  std::tie(q01, q2Vec) = programBuilder.ctrl(
      {q0, q1}, {q2}, [&](ValueRange targets) -> SmallVector<Value> {
        return SmallVector<Value>{programBuilder.x(targets[0])};
      });

  Value c2;
  std::tie(q2, c2) = programBuilder.measure(q2Vec[0]);

  programBuilder.sink(q01[0]);
  programBuilder.sink(q01[1]);
  programBuilder.sink(q2);
  program = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();
  auto r2 = referenceBuilder.allocQubit();
  r0 = referenceBuilder.h(r0);
  r1 = referenceBuilder.h(r1);
  r2 = referenceBuilder.h(r2);

  Value cr0;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  Value cr1;
  std::tie(r1, cr1) = referenceBuilder.measure(r1);

  auto andOp = arith::AndIOp::create(referenceBuilder, cr0, cr1);

  r2 = referenceBuilder.qcoIf(andOp.getResult(), r2, [&](Value qubit) -> Value {
    return referenceBuilder.x(qubit);
  });
  Value cr2;
  std::tie(r2, cr2) = referenceBuilder.measure(r2);
  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);
  referenceBuilder.sink(r2);

  reference = referenceBuilder.finalize({cr0, cr1, cr2});

  ASSERT_TRUE(runReplaceClassicalControlsPass(program.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/**
 * @brief Test: Tests replacing a classically controlled gate where two out of
 * three controls can be replaced.
 */
TEST_F(QCOReplaceClassicalControlsTest,
       replaceClassicalControlsTwoOfThreeControls) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type(),
       programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();
  auto q3 = programBuilder.allocQubit();
  q0 = programBuilder.h(q0);
  q1 = programBuilder.h(q1);
  q2 = programBuilder.h(q2);
  q3 = programBuilder.h(q3);

  Value c0;
  std::tie(q0, c0) = programBuilder.measure(q0);
  Value c1;
  std::tie(q1, c1) = programBuilder.measure(q1);

  SmallVector<Value> q012;
  SmallVector<Value> q3Vec;
  std::tie(q012, q3Vec) = programBuilder.ctrl(
      {q0, q1, q2}, {q3}, [&](ValueRange targets) -> SmallVector<Value> {
        return SmallVector<Value>{programBuilder.x(targets[0])};
      });

  Value c2;
  std::tie(q2, c2) = programBuilder.measure(q012[2]);
  Value c3;
  std::tie(q3, c3) = programBuilder.measure(q3Vec[0]);

  programBuilder.sink(q012[0]);
  programBuilder.sink(q012[1]);
  programBuilder.sink(q2);
  programBuilder.sink(q3);
  program = programBuilder.finalize({c0, c1, c2, c3});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type(),
       referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();
  auto r2 = referenceBuilder.allocQubit();
  auto r3 = referenceBuilder.allocQubit();
  r0 = referenceBuilder.h(r0);
  r1 = referenceBuilder.h(r1);
  r2 = referenceBuilder.h(r2);
  r3 = referenceBuilder.h(r3);

  Value cr0;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  Value cr1;
  std::tie(r1, cr1) = referenceBuilder.measure(r1);

  auto andOp = arith::AndIOp::create(referenceBuilder, cr0, cr1);

  SmallVector<Value> r23 =
      referenceBuilder.qcoIf(andOp.getResult(), {r2, r3},
                             [&](ValueRange qubits) -> SmallVector<Value> {
                               Value t2 = qubits[0];
                               Value t3 = qubits[1];
                               std::tie(t2, t3) = referenceBuilder.cx(t2, t3);
                               return SmallVector<Value>{t2, t3};
                             });
  Value cr2;
  std::tie(r2, cr2) = referenceBuilder.measure(r23[0]);
  Value cr3;
  std::tie(r3, cr3) = referenceBuilder.measure(r23[1]);
  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);
  referenceBuilder.sink(r2);
  referenceBuilder.sink(r3);

  reference = referenceBuilder.finalize({cr0, cr1, cr2, cr3});

  ASSERT_TRUE(runReplaceClassicalControlsPass(program.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/**
 * @brief Test: A measured target of a non-phase gate must not be mistaken for a
 * replaceable classical control.
 */
TEST_F(QCOReplaceClassicalControlsTest, doNotReplaceMeasuredNonPhaseTarget) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto target = programBuilder.h(programBuilder.allocQubit());
  auto control = programBuilder.h(programBuilder.allocQubit());

  Value initialTargetOutcome;
  std::tie(target, initialTargetOutcome) = programBuilder.measure(target);
  std::tie(control, target) = programBuilder.cx(control, target);

  Value controlOutcome;
  Value targetOutcome;
  std::tie(control, controlOutcome) = programBuilder.measure(control);
  std::tie(target, targetOutcome) = programBuilder.measure(target);
  programBuilder.sink(control);
  programBuilder.sink(target);
  program = programBuilder.finalize(
      {initialTargetOutcome, controlOutcome, targetOutcome});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto referenceTarget = referenceBuilder.h(referenceBuilder.allocQubit());
  auto referenceControl = referenceBuilder.h(referenceBuilder.allocQubit());

  Value referenceInitialTargetOutcome;
  std::tie(referenceTarget, referenceInitialTargetOutcome) =
      referenceBuilder.measure(referenceTarget);
  std::tie(referenceControl, referenceTarget) =
      referenceBuilder.cx(referenceControl, referenceTarget);

  Value referenceControlOutcome;
  Value referenceTargetOutcome;
  std::tie(referenceControl, referenceControlOutcome) =
      referenceBuilder.measure(referenceControl);
  std::tie(referenceTarget, referenceTargetOutcome) =
      referenceBuilder.measure(referenceTarget);
  referenceBuilder.sink(referenceControl);
  referenceBuilder.sink(referenceTarget);
  reference = referenceBuilder.finalize({referenceInitialTargetOutcome,
                                         referenceControlOutcome,
                                         referenceTargetOutcome});

  ASSERT_TRUE(runReplaceClassicalControlsPass(program.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/**
 * @brief Test: A measured control of a multi-target gate can be replaced
 * without attempting a single-target phase-gate swap.
 */
TEST_F(QCOReplaceClassicalControlsTest,
       replaceMeasuredControlOfMultiTargetGate) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto control = programBuilder.h(programBuilder.allocQubit());
  auto target0 = programBuilder.h(programBuilder.allocQubit());
  auto target1 = programBuilder.h(programBuilder.allocQubit());

  Value controlOutcome;
  std::tie(control, controlOutcome) = programBuilder.measure(control);
  auto [controlOut, targetsOut] =
      programBuilder.cswap(control, target0, target1);

  Value target0Outcome;
  Value target1Outcome;
  std::tie(target0, target0Outcome) = programBuilder.measure(targetsOut.first);
  std::tie(target1, target1Outcome) = programBuilder.measure(targetsOut.second);
  programBuilder.sink(controlOut);
  programBuilder.sink(target0);
  programBuilder.sink(target1);
  program =
      programBuilder.finalize({controlOutcome, target0Outcome, target1Outcome});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto referenceControl = referenceBuilder.h(referenceBuilder.allocQubit());
  auto referenceTarget0 = referenceBuilder.h(referenceBuilder.allocQubit());
  auto referenceTarget1 = referenceBuilder.h(referenceBuilder.allocQubit());

  Value referenceControlOutcome;
  std::tie(referenceControl, referenceControlOutcome) =
      referenceBuilder.measure(referenceControl);
  auto referenceTargets = referenceBuilder.qcoIf(
      referenceControlOutcome, {referenceTarget0, referenceTarget1},
      [&](ValueRange targets) -> SmallVector<Value> {
        auto [out0, out1] = referenceBuilder.swap(targets[0], targets[1]);
        return {out0, out1};
      });

  Value referenceTarget0Outcome;
  Value referenceTarget1Outcome;
  std::tie(referenceTarget0, referenceTarget0Outcome) =
      referenceBuilder.measure(referenceTargets[0]);
  std::tie(referenceTarget1, referenceTarget1Outcome) =
      referenceBuilder.measure(referenceTargets[1]);
  referenceBuilder.sink(referenceControl);
  referenceBuilder.sink(referenceTarget0);
  referenceBuilder.sink(referenceTarget1);
  reference = referenceBuilder.finalize({referenceControlOutcome,
                                         referenceTarget0Outcome,
                                         referenceTarget1Outcome});

  ASSERT_TRUE(runReplaceClassicalControlsPass(program.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/**
 * @brief Test: Tests replacing a classically controlled gate where a phase
 * target gate needs to be swapped with to achieve a replaceable control.
 */
TEST_F(QCOReplaceClassicalControlsTest, replaceClassicalControlsSwapPhase) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  q0 = programBuilder.h(q0);
  q1 = programBuilder.h(q1);

  Value c0;
  std::tie(q0, c0) = programBuilder.measure(q0);
  std::tie(q1, q0) = programBuilder.cz(q1, q0);
  Value c1;
  std::tie(q1, c1) = programBuilder.measure(q1);

  programBuilder.sink(q0);
  programBuilder.sink(q1);
  program = programBuilder.finalize({c0, c1});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();
  r0 = referenceBuilder.h(r0);
  r1 = referenceBuilder.h(r1);

  Value cr0;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);

  r1 = referenceBuilder.qcoIf(
      cr0, r1, [&](Value qubit) -> Value { return referenceBuilder.z(qubit); });
  Value cr1;
  std::tie(r1, cr1) = referenceBuilder.measure(r1);
  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);

  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runReplaceClassicalControlsPass(program.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/**
 * @brief Test: Tests that a phase target gate is not swapped with a
 * classical control if it's not necessary.
 */
TEST_F(QCOReplaceClassicalControlsTest,
       replaceClassicalControlsDontSwapPhaseIfNotNecessary) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  q0 = programBuilder.h(q0);
  q1 = programBuilder.h(q1);

  Value c0;
  std::tie(q0, c0) = programBuilder.measure(q0);
  Value c1;
  std::tie(q1, c1) = programBuilder.measure(q1);
  std::tie(q1, q0) = programBuilder.cz(q1, q0);
  Value c2;
  std::tie(q0, c2) = programBuilder.measure(q0);

  programBuilder.sink(q0);
  programBuilder.sink(q1);
  program = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               programBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();
  r0 = referenceBuilder.h(r0);
  r1 = referenceBuilder.h(r1);

  Value cr0;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  Value cr1;
  std::tie(r1, cr1) = referenceBuilder.measure(r1);

  r0 = referenceBuilder.qcoIf(cr1, r0, [&](Value qubits) -> Value {
    return referenceBuilder.z(qubits);
  });
  Value cr2;
  std::tie(r0, cr2) = referenceBuilder.measure(r0);
  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);

  reference = referenceBuilder.finalize({cr0, cr1, cr2});

  ASSERT_TRUE(runReplaceClassicalControlsPass(program.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/**
 * @brief Test: Tests replacing a classically controlled gate where one of two
 * control qubits of a phase gate is swapped with the target qubit.
 */
TEST_F(QCOReplaceClassicalControlsTest,
       replaceClassicalControlsSwapOneOfTwoPhase) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();
  q0 = programBuilder.h(q0);
  q1 = programBuilder.h(q1);
  q2 = programBuilder.h(q2);

  Value c0;
  std::tie(q0, c0) = programBuilder.measure(q0);
  SmallVector<Value> q12;
  SmallVector<Value> q0Vec;
  std::tie(q12, q0Vec) = programBuilder.ctrl(
      {q1, q2}, {q0}, [&](ValueRange targets) -> SmallVector<Value> {
        return SmallVector<Value>{programBuilder.z(targets[0])};
      });
  Value c1;
  std::tie(q1, c1) = programBuilder.measure(q12[0]);

  programBuilder.sink(q0Vec[0]);
  programBuilder.sink(q1);
  programBuilder.sink(q12[1]);
  program = programBuilder.finalize({c0, c1});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();
  auto r2 = referenceBuilder.allocQubit();
  r0 = referenceBuilder.h(r0);
  r1 = referenceBuilder.h(r1);
  r2 = referenceBuilder.h(r2);

  Value cr0;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);

  SmallVector<Value> r21 = referenceBuilder.qcoIf(
      cr0, {r2, r1}, [&](ValueRange qubits) -> SmallVector<Value> {
        Value t2 = qubits[0];
        Value t1 = qubits[1];
        std::tie(t2, t1) = referenceBuilder.cz(t2, t1);
        return SmallVector<Value>{t2, t1};
      });
  Value cr1;
  std::tie(r1, cr1) = referenceBuilder.measure(r21[1]);
  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);
  referenceBuilder.sink(r21[0]);

  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runReplaceClassicalControlsPass(program.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/**
 * @brief Test: Tests replacing a classically controlled gate where only one of
 * two controls can possibly be swapped with the target qubit of a phase
 * operation.
 */
TEST_F(QCOReplaceClassicalControlsTest,
       replaceClassicalControlsSwapOnlyPossiblePhase) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();
  q0 = programBuilder.h(q0);
  q1 = programBuilder.h(q1);
  q2 = programBuilder.h(q2);

  Value c0;
  std::tie(q0, c0) = programBuilder.measure(q0);
  Value c1;
  std::tie(q1, c1) = programBuilder.measure(q1);
  SmallVector<Value> q12;
  SmallVector<Value> q0Vec;
  std::tie(q12, q0Vec) = programBuilder.ctrl(
      {q1, q2}, {q0}, [&](ValueRange targets) -> SmallVector<Value> {
        return SmallVector<Value>{programBuilder.z(targets[0])};
      });
  Value c2;
  std::tie(q2, c2) = programBuilder.measure(q12[1]);

  programBuilder.sink(q0Vec[0]);
  programBuilder.sink(q12[0]);
  programBuilder.sink(q2);
  program = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();
  auto r2 = referenceBuilder.allocQubit();
  r0 = referenceBuilder.h(r0);
  r1 = referenceBuilder.h(r1);
  r2 = referenceBuilder.h(r2);

  Value cr0;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  Value cr1;
  std::tie(r1, cr1) = referenceBuilder.measure(r1);

  auto andOp = arith::AndIOp::create(referenceBuilder, cr1, cr0);

  r2 = referenceBuilder.qcoIf(andOp.getResult(), r2, [&](Value qubit) -> Value {
    return referenceBuilder.z(qubit);
  });
  Value cr2;
  std::tie(r2, cr2) = referenceBuilder.measure(r2);
  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);
  referenceBuilder.sink(r2);

  reference = referenceBuilder.finalize({cr0, cr1, cr2});

  ASSERT_TRUE(runReplaceClassicalControlsPass(program.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}
