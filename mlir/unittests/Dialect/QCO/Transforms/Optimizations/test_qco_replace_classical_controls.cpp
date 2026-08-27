/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Support/IRVerification.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
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

  static Value outcomeScaledAngle(QCOProgramBuilder& builder, Value outcome,
                                  const double theta, const double trueScale,
                                  const double falseScale) {
    Value thetaValue = builder.floatConstant(theta);
    Value trueValue = builder.floatConstant(trueScale);
    Value falseValue = builder.floatConstant(falseScale);
    Value scale = arith::SelectOp::create(builder, builder.getLoc(), outcome,
                                          trueValue, falseValue);
    return arith::MulFOp::create(builder, builder.getLoc(), thetaValue, scale);
  }
};

class QCOReplaceClassicalControlsRZZTest
    : public QCOReplaceClassicalControlsTest,
      public testing::WithParamInterface<size_t> {};

} // namespace

TEST(QCOClassicalControlPhaseIdentityTest,
     controlledRZAndRZZRewritesPreserveBasisPhases) {
  const auto expectSamePhase = [](const double actualExponent,
                                  const double expectedExponent) {
    const auto actual = std::polar(1.0, actualExponent);
    const auto expected = std::polar(1.0, expectedExponent);
    EXPECT_NEAR(actual.real(), expected.real(), 1e-12);
    EXPECT_NEAR(actual.imag(), expected.imag(), 1e-12);
  };

  constexpr std::array angles{0.125, -0.789, 2.3};
  for (const size_t numControls : {1U, 2U, 3U, 5U}) {
    const uint64_t controlMask = (uint64_t{1} << numControls) - 1U;
    const size_t numQubits = numControls + 2U;
    for (const double theta : angles) {
      for (uint64_t basis = 0; basis < (uint64_t{1} << numQubits); ++basis) {
        SCOPED_TRACE(testing::Message()
                     << "controls=" << numControls << ", theta=" << theta
                     << ", basis=" << basis);
        const bool controlsActive = (basis & controlMask) == controlMask;
        const bool targetA = (basis & (uint64_t{1} << numControls)) != 0;
        const bool targetB = (basis & (uint64_t{1} << (numControls + 1U))) != 0;

        double controlledRZExponent = 0.0;
        double rewrittenRZExponent = 0.0;
        if (controlsActive) {
          controlledRZExponent = targetA ? theta / 2.0 : -theta / 2.0;
          rewrittenRZExponent = targetA ? theta / 2.0 : -theta / 2.0;
        }
        expectSamePhase(rewrittenRZExponent, controlledRZExponent);

        const double zA = targetA ? -1.0 : 1.0;
        const double zB = targetB ? -1.0 : 1.0;
        const double controlledRZZExponent =
            controlsActive ? (-theta * zA * zB / 2.0) : 0.0;
        double oneMeasuredTargetExponent = 0.0;
        if (controlsActive) {
          const double selectedAngle = targetA ? -theta : theta;
          oneMeasuredTargetExponent = -selectedAngle * zB / 2.0;
        }
        expectSamePhase(oneMeasuredTargetExponent, controlledRZZExponent);

        double twoMeasuredTargetsExponent = 0.0;
        if (controlsActive) {
          twoMeasuredTargetsExponent =
              targetA != targetB ? theta / 2.0 : -theta / 2.0;
        }
        expectSamePhase(twoMeasuredTargetsExponent, controlledRZZExponent);
      }
    }
  }
}

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

TEST_F(QCOReplaceClassicalControlsTest,
       replaceMeasuredRZTargetWithSelectedPhase) {
  constexpr double theta = 0.789;
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto target = programBuilder.h(programBuilder.allocQubit());
  auto control = programBuilder.h(programBuilder.allocQubit());

  Value initialTargetOutcome;
  std::tie(target, initialTargetOutcome) = programBuilder.measure(target);
  std::tie(control, target) = programBuilder.crz(theta, control, target);
  Value controlOutcome;
  std::tie(control, controlOutcome) = programBuilder.measure(control);
  programBuilder.sink(control);
  programBuilder.sink(target);
  program = programBuilder.finalize({initialTargetOutcome, controlOutcome});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto referenceTarget = referenceBuilder.h(referenceBuilder.allocQubit());
  auto referenceControl = referenceBuilder.h(referenceBuilder.allocQubit());

  Value referenceInitialTargetOutcome;
  std::tie(referenceTarget, referenceInitialTargetOutcome) =
      referenceBuilder.measure(referenceTarget);
  Value selectedPhase = outcomeScaledAngle(
      referenceBuilder, referenceInitialTargetOutcome, theta, 0.5, -0.5);
  referenceControl = referenceBuilder.p(selectedPhase, referenceControl);
  Value referenceControlOutcome;
  std::tie(referenceControl, referenceControlOutcome) =
      referenceBuilder.measure(referenceControl);
  referenceBuilder.sink(referenceControl);
  referenceBuilder.sink(referenceTarget);
  reference = referenceBuilder.finalize(
      {referenceInitialTargetOutcome, referenceControlOutcome});

  ASSERT_TRUE(runReplaceClassicalControlsPass(*program).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(*reference).succeeded());
  EXPECT_TRUE(areModulesEquivalentWithPermutations(*program, *reference));
}

TEST_F(QCOReplaceClassicalControlsTest,
       replaceMeasuredRZTargetWithMultiControlPhase) {
  constexpr double theta = 0.789;
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto q = programBuilder.allocQubitRegister(3);
  for (auto& qubit : q.qubits) {
    qubit = programBuilder.h(qubit);
  }
  Value targetOutcome;
  std::tie(q[2], targetOutcome) = programBuilder.measure(q[2]);
  auto [controls, target] = programBuilder.mcrz(theta, {q[0], q[1]}, q[2]);
  Value control0;
  Value control0Outcome;
  std::tie(control0, control0Outcome) = programBuilder.measure(controls[0]);
  Value control1;
  Value control1Outcome;
  std::tie(control1, control1Outcome) = programBuilder.measure(controls[1]);
  programBuilder.sink(control0);
  programBuilder.sink(control1);
  programBuilder.sink(target);
  program = programBuilder.finalize(
      {targetOutcome, control0Outcome, control1Outcome});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r = referenceBuilder.allocQubitRegister(3);
  for (auto& qubit : r.qubits) {
    qubit = referenceBuilder.h(qubit);
  }
  Value referenceTargetOutcome;
  std::tie(r[2], referenceTargetOutcome) = referenceBuilder.measure(r[2]);
  Value selectedPhase = outcomeScaledAngle(
      referenceBuilder, referenceTargetOutcome, theta, 0.5, -0.5);
  std::tie(r[0], r[1]) = referenceBuilder.cp(selectedPhase, r[0], r[1]);
  Value referenceControl0Outcome;
  std::tie(r[0], referenceControl0Outcome) = referenceBuilder.measure(r[0]);
  Value referenceControl1Outcome;
  std::tie(r[1], referenceControl1Outcome) = referenceBuilder.measure(r[1]);
  for (auto qubit : r.qubits) {
    referenceBuilder.sink(qubit);
  }
  reference = referenceBuilder.finalize({referenceTargetOutcome,
                                         referenceControl0Outcome,
                                         referenceControl1Outcome});

  ASSERT_TRUE(runReplaceClassicalControlsPass(*program).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(*reference).succeeded());
  EXPECT_TRUE(areModulesEquivalentWithPermutations(*program, *reference));
}

TEST_F(QCOReplaceClassicalControlsTest,
       removesRZWhenControlAndTargetAreMeasured) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type(),
       programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto control = programBuilder.h(programBuilder.allocQubit());
  auto target = programBuilder.h(programBuilder.allocQubit());
  Value controlOutcome;
  Value targetOutcome;
  std::tie(control, controlOutcome) = programBuilder.measure(control);
  std::tie(target, targetOutcome) = programBuilder.measure(target);
  std::tie(control, target) = programBuilder.crz(0.789, control, target);
  Value outputControlOutcome;
  Value outputTargetOutcome;
  std::tie(control, outputControlOutcome) = programBuilder.measure(control);
  std::tie(target, outputTargetOutcome) = programBuilder.measure(target);
  programBuilder.sink(control);
  programBuilder.sink(target);
  program =
      programBuilder.finalize({controlOutcome, targetOutcome,
                               outputControlOutcome, outputTargetOutcome});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type(),
       referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto referenceControl = referenceBuilder.h(referenceBuilder.allocQubit());
  auto referenceTarget = referenceBuilder.h(referenceBuilder.allocQubit());
  Value referenceControlOutcome;
  Value referenceTargetOutcome;
  std::tie(referenceControl, referenceControlOutcome) =
      referenceBuilder.measure(referenceControl);
  std::tie(referenceTarget, referenceTargetOutcome) =
      referenceBuilder.measure(referenceTarget);
  Value referenceOutputControlOutcome;
  Value referenceOutputTargetOutcome;
  std::tie(referenceControl, referenceOutputControlOutcome) =
      referenceBuilder.measure(referenceControl);
  std::tie(referenceTarget, referenceOutputTargetOutcome) =
      referenceBuilder.measure(referenceTarget);
  referenceBuilder.sink(referenceControl);
  referenceBuilder.sink(referenceTarget);
  reference = referenceBuilder.finalize(
      {referenceControlOutcome, referenceTargetOutcome,
       referenceOutputControlOutcome, referenceOutputTargetOutcome});

  ASSERT_TRUE(runReplaceClassicalControlsPass(*program).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(*reference).succeeded());
  EXPECT_TRUE(areModulesEquivalentWithPermutations(*program, *reference));
}

TEST_P(QCOReplaceClassicalControlsRZZTest,
       replaceMeasuredRZZTargetWithSelectedAngle) {
  constexpr double theta = 0.789;
  const size_t measuredTargetIndex = GetParam();
  const size_t otherTargetIndex = 1U - measuredTargetIndex;

  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto control = programBuilder.h(programBuilder.allocQubit());
  std::array targets{programBuilder.h(programBuilder.allocQubit()),
                     programBuilder.h(programBuilder.allocQubit())};
  Value measuredTargetOutcome;
  std::tie(targets[measuredTargetIndex], measuredTargetOutcome) =
      programBuilder.measure(targets[measuredTargetIndex]);
  auto [outputControl, outputTargetPair] =
      programBuilder.crzz(theta, control, targets[0], targets[1]);
  std::array outputTargets{outputTargetPair.first, outputTargetPair.second};
  Value controlOutcome;
  std::tie(outputControl, controlOutcome) =
      programBuilder.measure(outputControl);
  Value otherTargetOutcome;
  std::tie(outputTargets[otherTargetIndex], otherTargetOutcome) =
      programBuilder.measure(outputTargets[otherTargetIndex]);
  programBuilder.sink(outputControl);
  programBuilder.sink(outputTargets[0]);
  programBuilder.sink(outputTargets[1]);
  program = programBuilder.finalize(
      {measuredTargetOutcome, controlOutcome, otherTargetOutcome});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto referenceControl = referenceBuilder.h(referenceBuilder.allocQubit());
  std::array referenceTargets{
      referenceBuilder.h(referenceBuilder.allocQubit()),
      referenceBuilder.h(referenceBuilder.allocQubit())};
  Value referenceMeasuredTargetOutcome;
  std::tie(referenceTargets[measuredTargetIndex],
           referenceMeasuredTargetOutcome) =
      referenceBuilder.measure(referenceTargets[measuredTargetIndex]);
  Value selectedAngle = outcomeScaledAngle(
      referenceBuilder, referenceMeasuredTargetOutcome, theta, -1.0, 1.0);
  std::tie(referenceControl, referenceTargets[otherTargetIndex]) =
      referenceBuilder.crz(selectedAngle, referenceControl,
                           referenceTargets[otherTargetIndex]);
  Value referenceControlOutcome;
  std::tie(referenceControl, referenceControlOutcome) =
      referenceBuilder.measure(referenceControl);
  Value referenceOtherTargetOutcome;
  std::tie(referenceTargets[otherTargetIndex], referenceOtherTargetOutcome) =
      referenceBuilder.measure(referenceTargets[otherTargetIndex]);
  referenceBuilder.sink(referenceControl);
  referenceBuilder.sink(referenceTargets[0]);
  referenceBuilder.sink(referenceTargets[1]);
  reference = referenceBuilder.finalize({referenceMeasuredTargetOutcome,
                                         referenceControlOutcome,
                                         referenceOtherTargetOutcome});

  ASSERT_TRUE(runReplaceClassicalControlsPass(*program).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(*reference).succeeded());
  EXPECT_TRUE(areModulesEquivalentWithPermutations(*program, *reference));
}

TEST_F(QCOReplaceClassicalControlsTest,
       replaceMeasuredRZZTargetPreservesReversedYields) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto control = programBuilder.h(programBuilder.allocQubit());
  auto target0 = programBuilder.h(programBuilder.allocQubit());
  auto target1 = programBuilder.h(programBuilder.allocQubit());
  Value measuredTargetOutcome;
  std::tie(target0, measuredTargetOutcome) = programBuilder.measure(target0);
  Value measuredTarget = target0;
  auto [controls, targets] =
      programBuilder.ctrl({control}, {target0, target1},
                          [&](ValueRange args) -> SmallVector<Value> {
                            auto [output0, output1] =
                                programBuilder.rzz(0.789, args[0], args[1]);
                            return {output1, output0};
                          });
  Value output0Outcome;
  std::tie(target0, output0Outcome) = programBuilder.measure(targets[0]);
  auto output0Measurement = cast<MeasureOp>(target0.getDefiningOp());
  Value output1Outcome;
  std::tie(target1, output1Outcome) = programBuilder.measure(targets[1]);
  auto output1Measurement = cast<MeasureOp>(target1.getDefiningOp());
  programBuilder.sink(controls[0]);
  programBuilder.sink(target0);
  programBuilder.sink(target1);
  program = programBuilder.finalize(
      {measuredTargetOutcome, output0Outcome, output1Outcome});

  ASSERT_TRUE(runReplaceClassicalControlsPass(*program).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());
  EXPECT_NE(output0Measurement.getQubitIn(), measuredTarget);
  EXPECT_EQ(output1Measurement.getQubitIn(), measuredTarget);
}

TEST_F(QCOReplaceClassicalControlsTest,
       doesNotReplaceMeasuredRZZTargetWithAdditionalTarget) {
  programBuilder.initialize({programBuilder.getI1Type()});
  auto control = programBuilder.h(programBuilder.allocQubit());
  auto target0 = programBuilder.h(programBuilder.allocQubit());
  auto target1 = programBuilder.h(programBuilder.allocQubit());
  auto passthrough = programBuilder.h(programBuilder.allocQubit());
  Value outcome;
  std::tie(target0, outcome) = programBuilder.measure(target0);
  auto [controls, targets] =
      programBuilder.ctrl({control}, {target0, target1, passthrough},
                          [&](ValueRange args) -> SmallVector<Value> {
                            auto [output0, output1] =
                                programBuilder.rzz(0.789, args[0], args[1]);
                            return {output0, output1, args[2]};
                          });
  programBuilder.sink(controls[0]);
  programBuilder.sink(targets[0]);
  programBuilder.sink(targets[1]);
  programBuilder.sink(targets[2]);
  program = programBuilder.finalize({outcome});

  ASSERT_TRUE(runReplaceClassicalControlsPass(*program).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());
  size_t ifCount = 0;
  program->walk([&](IfOp) { ++ifCount; });
  EXPECT_EQ(ifCount, 0U);
}

TEST_F(QCOReplaceClassicalControlsTest,
       replacesMeasuredRZZTargetWhenAllControlsAreMeasured) {
  constexpr double theta = 0.789;
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto control = programBuilder.h(programBuilder.allocQubit());
  auto target0 = programBuilder.h(programBuilder.allocQubit());
  auto target1 = programBuilder.h(programBuilder.allocQubit());
  Value controlOutcome;
  std::tie(control, controlOutcome) = programBuilder.measure(control);
  Value targetOutcome;
  std::tie(target0, targetOutcome) = programBuilder.measure(target0);
  auto [outputControl, outputTargets] =
      programBuilder.crzz(theta, control, target0, target1);
  Value otherTargetOutcome;
  std::tie(target1, otherTargetOutcome) =
      programBuilder.measure(outputTargets.second);
  programBuilder.sink(outputControl);
  programBuilder.sink(outputTargets.first);
  programBuilder.sink(target1);
  program = programBuilder.finalize(
      {controlOutcome, targetOutcome, otherTargetOutcome});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto referenceControl = referenceBuilder.h(referenceBuilder.allocQubit());
  auto referenceTarget0 = referenceBuilder.h(referenceBuilder.allocQubit());
  auto referenceTarget1 = referenceBuilder.h(referenceBuilder.allocQubit());
  Value referenceControlOutcome;
  std::tie(referenceControl, referenceControlOutcome) =
      referenceBuilder.measure(referenceControl);
  Value referenceTargetOutcome;
  std::tie(referenceTarget0, referenceTargetOutcome) =
      referenceBuilder.measure(referenceTarget0);
  Value selectedAngle = outcomeScaledAngle(
      referenceBuilder, referenceTargetOutcome, theta, -1.0, 1.0);
  referenceTarget1 = referenceBuilder.qcoIf(
      referenceControlOutcome, referenceTarget1,
      [&](Value target) { return referenceBuilder.rz(selectedAngle, target); });
  Value referenceOtherTargetOutcome;
  std::tie(referenceTarget1, referenceOtherTargetOutcome) =
      referenceBuilder.measure(referenceTarget1);
  referenceBuilder.sink(referenceControl);
  referenceBuilder.sink(referenceTarget0);
  referenceBuilder.sink(referenceTarget1);
  reference = referenceBuilder.finalize({referenceControlOutcome,
                                         referenceTargetOutcome,
                                         referenceOtherTargetOutcome});

  ASSERT_TRUE(runReplaceClassicalControlsPass(*program).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(*reference).succeeded());
  EXPECT_TRUE(areModulesEquivalentWithPermutations(*program, *reference));
}

TEST_F(QCOReplaceClassicalControlsTest, replacesRZZWhenBothTargetsAreMeasured) {
  constexpr double theta = 0.789;
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto control = programBuilder.h(programBuilder.allocQubit());
  auto target0 = programBuilder.h(programBuilder.allocQubit());
  auto target1 = programBuilder.h(programBuilder.allocQubit());
  Value outcome0;
  Value outcome1;
  std::tie(target0, outcome0) = programBuilder.measure(target0);
  std::tie(target1, outcome1) = programBuilder.measure(target1);
  auto [outputControl, outputTargets] =
      programBuilder.crzz(theta, control, target0, target1);
  Value controlOutcome;
  std::tie(control, controlOutcome) = programBuilder.measure(outputControl);
  programBuilder.sink(control);
  programBuilder.sink(outputTargets.first);
  programBuilder.sink(outputTargets.second);
  program = programBuilder.finalize({outcome0, outcome1, controlOutcome});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto referenceControl = referenceBuilder.h(referenceBuilder.allocQubit());
  auto referenceTarget0 = referenceBuilder.h(referenceBuilder.allocQubit());
  auto referenceTarget1 = referenceBuilder.h(referenceBuilder.allocQubit());
  Value referenceOutcome0;
  Value referenceOutcome1;
  std::tie(referenceTarget0, referenceOutcome0) =
      referenceBuilder.measure(referenceTarget0);
  std::tie(referenceTarget1, referenceOutcome1) =
      referenceBuilder.measure(referenceTarget1);
  Value outcomesDiffer =
      arith::XOrIOp::create(referenceBuilder, referenceBuilder.getLoc(),
                            referenceOutcome0, referenceOutcome1);
  Value selectedPhase =
      outcomeScaledAngle(referenceBuilder, outcomesDiffer, theta, 0.5, -0.5);
  referenceControl = referenceBuilder.p(selectedPhase, referenceControl);
  Value referenceControlOutcome;
  std::tie(referenceControl, referenceControlOutcome) =
      referenceBuilder.measure(referenceControl);
  referenceBuilder.sink(referenceControl);
  referenceBuilder.sink(referenceTarget0);
  referenceBuilder.sink(referenceTarget1);
  reference = referenceBuilder.finalize(
      {referenceOutcome0, referenceOutcome1, referenceControlOutcome});

  ASSERT_TRUE(runReplaceClassicalControlsPass(*program).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(*reference).succeeded());
  EXPECT_TRUE(areModulesEquivalentWithPermutations(*program, *reference));
}

TEST_F(QCOReplaceClassicalControlsTest, removesRZZWhenAllQubitsAreMeasured) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type(),
       programBuilder.getI1Type(), programBuilder.getI1Type(),
       programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto control = programBuilder.h(programBuilder.allocQubit());
  auto target0 = programBuilder.h(programBuilder.allocQubit());
  auto target1 = programBuilder.h(programBuilder.allocQubit());
  Value controlOutcome;
  Value target0Outcome;
  Value target1Outcome;
  std::tie(control, controlOutcome) = programBuilder.measure(control);
  std::tie(target0, target0Outcome) = programBuilder.measure(target0);
  std::tie(target1, target1Outcome) = programBuilder.measure(target1);
  auto [outputControl, outputTargets] =
      programBuilder.crzz(0.789, control, target0, target1);
  control = outputControl;
  target0 = outputTargets.first;
  target1 = outputTargets.second;
  Value outputControlOutcome;
  Value outputTarget0Outcome;
  Value outputTarget1Outcome;
  std::tie(control, outputControlOutcome) = programBuilder.measure(control);
  std::tie(target0, outputTarget0Outcome) = programBuilder.measure(target0);
  std::tie(target1, outputTarget1Outcome) = programBuilder.measure(target1);
  programBuilder.sink(control);
  programBuilder.sink(target0);
  programBuilder.sink(target1);
  program = programBuilder.finalize(
      {controlOutcome, target0Outcome, target1Outcome, outputControlOutcome,
       outputTarget0Outcome, outputTarget1Outcome});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type(),
       referenceBuilder.getI1Type(), referenceBuilder.getI1Type(),
       referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto referenceControl = referenceBuilder.h(referenceBuilder.allocQubit());
  auto referenceTarget0 = referenceBuilder.h(referenceBuilder.allocQubit());
  auto referenceTarget1 = referenceBuilder.h(referenceBuilder.allocQubit());
  Value referenceControlOutcome;
  Value referenceTarget0Outcome;
  Value referenceTarget1Outcome;
  std::tie(referenceControl, referenceControlOutcome) =
      referenceBuilder.measure(referenceControl);
  std::tie(referenceTarget0, referenceTarget0Outcome) =
      referenceBuilder.measure(referenceTarget0);
  std::tie(referenceTarget1, referenceTarget1Outcome) =
      referenceBuilder.measure(referenceTarget1);
  Value referenceOutputControlOutcome;
  Value referenceOutputTarget0Outcome;
  Value referenceOutputTarget1Outcome;
  std::tie(referenceControl, referenceOutputControlOutcome) =
      referenceBuilder.measure(referenceControl);
  std::tie(referenceTarget0, referenceOutputTarget0Outcome) =
      referenceBuilder.measure(referenceTarget0);
  std::tie(referenceTarget1, referenceOutputTarget1Outcome) =
      referenceBuilder.measure(referenceTarget1);
  referenceBuilder.sink(referenceControl);
  referenceBuilder.sink(referenceTarget0);
  referenceBuilder.sink(referenceTarget1);
  reference = referenceBuilder.finalize(
      {referenceControlOutcome, referenceTarget0Outcome,
       referenceTarget1Outcome, referenceOutputControlOutcome,
       referenceOutputTarget0Outcome, referenceOutputTarget1Outcome});

  ASSERT_TRUE(runReplaceClassicalControlsPass(*program).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(*reference).succeeded());
  EXPECT_TRUE(areModulesEquivalentWithPermutations(*program, *reference));
}

TEST_F(QCOReplaceClassicalControlsTest,
       replacesMeasuredRZZControlWithoutMeasuredTargets) {
  programBuilder.initialize({programBuilder.getI1Type()});
  auto control = programBuilder.h(programBuilder.allocQubit());
  auto target0 = programBuilder.h(programBuilder.allocQubit());
  auto target1 = programBuilder.h(programBuilder.allocQubit());
  Value outcome;
  std::tie(control, outcome) = programBuilder.measure(control);
  auto [outputControl, outputTargets] =
      programBuilder.crzz(0.789, control, target0, target1);
  programBuilder.sink(outputControl);
  programBuilder.sink(outputTargets.first);
  programBuilder.sink(outputTargets.second);
  program = programBuilder.finalize({outcome});

  ASSERT_TRUE(runReplaceClassicalControlsPass(*program).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());
  EXPECT_FALSE(program->getBody()
                   ->walk([](CtrlOp ctrlOp) {
                     return llvm::any_of(ctrlOp.getControlsIn(),
                                         [](Value input) {
                                           return isa_and_nonnull<MeasureOp>(
                                               input.getDefiningOp());
                                         })
                                ? WalkResult::interrupt()
                                : WalkResult::advance();
                   })
                   .wasInterrupted());
}

INSTANTIATE_TEST_SUITE_P(MeasuredTargetPositions,
                         QCOReplaceClassicalControlsRZZTest,
                         testing::Values(0U, 1U));

TEST_F(QCOReplaceClassicalControlsTest,
       replaceMeasuredRZZTargetWithMultipleControls) {
  constexpr double theta = 0.789;
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto q = programBuilder.allocQubitRegister(4);
  for (auto& qubit : q.qubits) {
    qubit = programBuilder.h(qubit);
  }
  Value measuredTargetOutcome;
  std::tie(q[2], measuredTargetOutcome) = programBuilder.measure(q[2]);
  auto [controls, targets] =
      programBuilder.mcrzz(theta, {q[0], q[1]}, q[2], q[3]);
  Value measuredControl;
  Value controlOutcome;
  std::tie(measuredControl, controlOutcome) =
      programBuilder.measure(controls[0]);
  Value otherTargetOutcome;
  std::tie(targets.second, otherTargetOutcome) =
      programBuilder.measure(targets.second);
  programBuilder.sink(measuredControl);
  programBuilder.sink(controls[1]);
  programBuilder.sink(targets.first);
  programBuilder.sink(targets.second);
  program = programBuilder.finalize(
      {measuredTargetOutcome, controlOutcome, otherTargetOutcome});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r = referenceBuilder.allocQubitRegister(4);
  for (auto& qubit : r.qubits) {
    qubit = referenceBuilder.h(qubit);
  }
  Value referenceMeasuredTargetOutcome;
  std::tie(r[2], referenceMeasuredTargetOutcome) =
      referenceBuilder.measure(r[2]);
  Value selectedAngle = outcomeScaledAngle(
      referenceBuilder, referenceMeasuredTargetOutcome, theta, -1.0, 1.0);
  ValueRange referenceControls;
  std::tie(referenceControls, r[3]) =
      referenceBuilder.mcrz(selectedAngle, {r[0], r[1]}, r[3]);
  r[0] = referenceControls[0];
  r[1] = referenceControls[1];
  Value referenceControlOutcome;
  std::tie(r[0], referenceControlOutcome) = referenceBuilder.measure(r[0]);
  Value referenceOtherTargetOutcome;
  std::tie(r[3], referenceOtherTargetOutcome) = referenceBuilder.measure(r[3]);
  for (auto qubit : r.qubits) {
    referenceBuilder.sink(qubit);
  }
  reference = referenceBuilder.finalize({referenceMeasuredTargetOutcome,
                                         referenceControlOutcome,
                                         referenceOtherTargetOutcome});

  ASSERT_TRUE(runReplaceClassicalControlsPass(*program).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(*reference).succeeded());
  EXPECT_TRUE(areModulesEquivalentWithPermutations(*program, *reference));
}

TEST_F(QCOReplaceClassicalControlsTest,
       replaceMeasuredRZZTargetAndMeasuredControl) {
  constexpr double theta = 0.789;
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type(),
       programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q = programBuilder.allocQubitRegister(4);
  for (auto& qubit : q.qubits) {
    qubit = programBuilder.h(qubit);
  }
  Value controlOutcome;
  std::tie(q[0], controlOutcome) = programBuilder.measure(q[0]);
  Value targetOutcome;
  std::tie(q[2], targetOutcome) = programBuilder.measure(q[2]);
  auto [controls, targets] =
      programBuilder.mcrzz(theta, {q[0], q[1]}, q[2], q[3]);
  Value quantumControl = controls[1];
  Value quantumControlOutcome;
  std::tie(quantumControl, quantumControlOutcome) =
      programBuilder.measure(quantumControl);
  Value otherTargetOutcome;
  std::tie(targets.second, otherTargetOutcome) =
      programBuilder.measure(targets.second);
  programBuilder.sink(controls[0]);
  programBuilder.sink(quantumControl);
  programBuilder.sink(targets.first);
  programBuilder.sink(targets.second);
  program =
      programBuilder.finalize({controlOutcome, targetOutcome,
                               quantumControlOutcome, otherTargetOutcome});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type(),
       referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r = referenceBuilder.allocQubitRegister(4);
  for (auto& qubit : r.qubits) {
    qubit = referenceBuilder.h(qubit);
  }
  Value referenceControlOutcome;
  std::tie(r[0], referenceControlOutcome) = referenceBuilder.measure(r[0]);
  Value referenceTargetOutcome;
  std::tie(r[2], referenceTargetOutcome) = referenceBuilder.measure(r[2]);
  Value selectedAngle = outcomeScaledAngle(
      referenceBuilder, referenceTargetOutcome, theta, -1.0, 1.0);
  auto conditionalQubits = referenceBuilder.qcoIf(
      referenceControlOutcome, ValueRange{r[1], r[3]},
      [&](ValueRange qubits) -> SmallVector<Value> {
        Value quantumControl = qubits[0];
        Value otherTarget = qubits[1];
        std::tie(quantumControl, otherTarget) =
            referenceBuilder.crz(selectedAngle, quantumControl, otherTarget);
        return {quantumControl, otherTarget};
      });
  r[1] = conditionalQubits[0];
  r[3] = conditionalQubits[1];
  Value referenceQuantumControlOutcome;
  std::tie(r[1], referenceQuantumControlOutcome) =
      referenceBuilder.measure(r[1]);
  Value referenceOtherTargetOutcome;
  std::tie(r[3], referenceOtherTargetOutcome) = referenceBuilder.measure(r[3]);
  for (Value qubit : r.qubits) {
    referenceBuilder.sink(qubit);
  }
  reference = referenceBuilder.finalize(
      {referenceControlOutcome, referenceTargetOutcome,
       referenceQuantumControlOutcome, referenceOtherTargetOutcome});

  ASSERT_TRUE(runReplaceClassicalControlsPass(*program).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(*reference).succeeded());
  EXPECT_TRUE(areModulesEquivalentWithPermutations(*program, *reference));
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
