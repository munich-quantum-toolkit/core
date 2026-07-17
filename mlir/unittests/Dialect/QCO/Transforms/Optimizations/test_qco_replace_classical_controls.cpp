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

#include <tuple>

namespace {

using namespace mlir;
using namespace mlir::qco;

class QCOReplaceClassicalControlsTest : public testing::Test {

protected:
  MLIRContext context;
  QCOProgramBuilder programBuilder;
  QCOProgramBuilder referenceBuilder;
  OwningOpRef<ModuleOp> module;
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
  runReplaceClassicalControlsPass(ModuleOp module,
                                  bool liftMeasurements = false) {
    PassManager pm(module.getContext());
    pm.addPass(createReplaceClassicalControls());
    if (liftMeasurements) {
      pm.addPass(createMeasurementLifting());
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

TEST_F(QCOReplaceClassicalControlsTest, replaceClassicalControlsOnlyControl) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();

  Value c0;
  std::tie(q0, c0) = programBuilder.measure(q0);
  std::tie(q0, q1) = programBuilder.cx(q0, q1);
  Value c1;
  std::tie(q1, c1) = programBuilder.measure(q1);

  programBuilder.sink(q0);
  programBuilder.sink(q1);
  module = programBuilder.finalize({c0, c1});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();

  Value cr0;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);

  r1 = referenceBuilder.qcoIf(
      cr0, {r1}, [&](ValueRange qubits) -> SmallVector<Value> {
        return SmallVector<Value>{referenceBuilder.x(qubits[0])};
      })[0];
  Value cr1;
  std::tie(r1, cr1) = referenceBuilder.measure(r1);
  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);

  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runReplaceClassicalControlsPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOReplaceClassicalControlsTest,
       replaceClassicalControlsOneOfTwoControls) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();

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
  module = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();
  auto r2 = referenceBuilder.allocQubit();

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

  ASSERT_TRUE(runReplaceClassicalControlsPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOReplaceClassicalControlsTest,
       replaceClassicalControlsTwoOfTwoControls) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();

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
  module = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();
  auto r2 = referenceBuilder.allocQubit();

  Value cr0;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  Value cr1;
  std::tie(r1, cr1) = referenceBuilder.measure(r1);

  auto andOp = arith::AndIOp::create(referenceBuilder, cr0, cr1);

  r2 = referenceBuilder.qcoIf(
      andOp.getResult(), {r2}, [&](ValueRange qubits) -> SmallVector<Value> {
        return SmallVector<Value>{referenceBuilder.x(qubits[0])};
      })[0];
  Value cr2;
  std::tie(r2, cr2) = referenceBuilder.measure(r2);
  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);
  referenceBuilder.sink(r2);

  reference = referenceBuilder.finalize({cr0, cr1, cr2});

  ASSERT_TRUE(runReplaceClassicalControlsPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOReplaceClassicalControlsTest,
       replaceClassicalControlsTwoOfThreeControls) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type(),
       programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();
  auto q3 = programBuilder.allocQubit();

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
  module = programBuilder.finalize({c0, c1, c2, c3});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type(),
       referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();
  auto r2 = referenceBuilder.allocQubit();
  auto r3 = referenceBuilder.allocQubit();

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

  ASSERT_TRUE(runReplaceClassicalControlsPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOReplaceClassicalControlsTest, replaceClassicalControlsSwapDiagonal) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();

  Value c0;
  std::tie(q0, c0) = programBuilder.measure(q0);
  std::tie(q1, q0) = programBuilder.cz(q1, q0);
  Value c1;
  std::tie(q1, c1) = programBuilder.measure(q1);

  programBuilder.sink(q0);
  programBuilder.sink(q1);
  module = programBuilder.finalize({c0, c1});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();

  Value cr0;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);

  r1 = referenceBuilder.qcoIf(
      cr0, {r1}, [&](ValueRange qubits) -> SmallVector<Value> {
        return SmallVector<Value>{referenceBuilder.z(qubits[0])};
      })[0];
  Value cr1;
  std::tie(r1, cr1) = referenceBuilder.measure(r1);
  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);

  reference = referenceBuilder.finalize({cr0, cr1});

  ASSERT_TRUE(runReplaceClassicalControlsPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOReplaceClassicalControlsTest,
       replaceClassicalControlsDontSwapDiagonalIfNotNecessary) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();

  Value c0;
  std::tie(q0, c0) = programBuilder.measure(q0);
  Value c1;
  std::tie(q1, c1) = programBuilder.measure(q1);
  std::tie(q1, q0) = programBuilder.cz(q1, q0);
  Value c2;
  std::tie(q0, c2) = programBuilder.measure(q0);

  programBuilder.sink(q0);
  programBuilder.sink(q1);
  module = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               programBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();

  Value cr0;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  Value cr1;
  std::tie(r1, cr1) = referenceBuilder.measure(r1);

  r0 = referenceBuilder.qcoIf(
      cr1, {r0}, [&](ValueRange qubits) -> SmallVector<Value> {
        return SmallVector<Value>{referenceBuilder.z(qubits[0])};
      })[0];
  Value cr2;
  std::tie(r0, cr2) = referenceBuilder.measure(r0);
  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);

  reference = referenceBuilder.finalize({cr0, cr1, cr2});

  ASSERT_TRUE(runReplaceClassicalControlsPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOReplaceClassicalControlsTest,
       replaceClassicalControlsSwapOneOfTwoDiagonal) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();

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
  module = programBuilder.finalize({c0, c1});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();
  auto r2 = referenceBuilder.allocQubit();

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

  ASSERT_TRUE(runReplaceClassicalControlsPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOReplaceClassicalControlsTest,
       replaceClassicalControlsSwapOnlyPossibleDiagonal) {
  programBuilder.initialize({programBuilder.getI1Type(),
                             programBuilder.getI1Type(),
                             programBuilder.getI1Type()});
  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();

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
  module = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r0 = referenceBuilder.allocQubit();
  auto r1 = referenceBuilder.allocQubit();
  auto r2 = referenceBuilder.allocQubit();

  Value cr0;
  std::tie(r0, cr0) = referenceBuilder.measure(r0);
  Value cr1;
  std::tie(r1, cr1) = referenceBuilder.measure(r1);

  auto andOp = arith::AndIOp::create(referenceBuilder, cr1, cr0);

  r2 = referenceBuilder.qcoIf(
      andOp.getResult(), {r2}, [&](ValueRange qubits) -> SmallVector<Value> {
        return SmallVector<Value>{referenceBuilder.z(qubits[0])};
      })[0];
  Value cr2;
  std::tie(r2, cr2) = referenceBuilder.measure(r2);
  referenceBuilder.sink(r0);
  referenceBuilder.sink(r1);
  referenceBuilder.sink(r2);

  reference = referenceBuilder.finalize({cr0, cr1, cr2});

  ASSERT_TRUE(runReplaceClassicalControlsPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}
