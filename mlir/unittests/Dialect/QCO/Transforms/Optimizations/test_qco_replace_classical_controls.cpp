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
  auto q0S0 = programBuilder.allocQubit();
  auto q1S0 = programBuilder.allocQubit();

  auto [q0S1, c0] = programBuilder.measure(q0S0);
  auto [q0S2, q1S1] = programBuilder.cx(q0S1, q1S0);
  auto [q1S2, c1] = programBuilder.measure(q1S1);

  programBuilder.sink(q0S2);
  programBuilder.sink(q1S2);
  module = programBuilder.finalize({c0, c1});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0S0 = referenceBuilder.allocQubit();
  auto r1S0 = referenceBuilder.allocQubit();

  auto [r0S1, cr0] = referenceBuilder.measure(r0S0);

  auto r1S1 = referenceBuilder.qcoIf(
      cr0, {r1S0}, [&](ValueRange qubits) -> SmallVector<Value> {
        auto q1Then = referenceBuilder.x(qubits[0]);
        return SmallVector<Value>{q1Then};
      })[0];
  auto [r1S2, cr1] = referenceBuilder.measure(r1S1);
  referenceBuilder.sink(r0S1);
  referenceBuilder.sink(r1S2);

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
  auto q0S0 = programBuilder.allocQubit();
  auto q1S0 = programBuilder.allocQubit();
  auto q2S0 = programBuilder.allocQubit();

  auto [q0S1, c0] = programBuilder.measure(q0S0);

  auto [q01S1, q2S1] =
      programBuilder.ctrl({q0S1, q1S0}, {q2S0},
                          [&](const ValueRange targets) -> SmallVector<Value> {
                            auto q = programBuilder.x(targets[0]);
                            return SmallVector<Value>{q};
                          });

  auto [q1S2, c1] = programBuilder.measure(q01S1[1]);
  auto [q2S2, c2] = programBuilder.measure(q2S1[0]);

  programBuilder.sink(q01S1[0]);
  programBuilder.sink(q1S2);
  programBuilder.sink(q2S2);
  module = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r0S0 = referenceBuilder.allocQubit();
  auto r1S0 = referenceBuilder.allocQubit();
  auto r2S0 = referenceBuilder.allocQubit();

  auto [r0S1, cr0] = referenceBuilder.measure(r0S0);

  auto r12 = referenceBuilder.qcoIf(
      cr0, {r1S0, r2S0}, [&](ValueRange qubits) -> SmallVector<Value> {
        auto [r1, r2] = referenceBuilder.cx(qubits[0], qubits[1]);
        return SmallVector<Value>{r1, r2};
      });
  auto [r1S2, cr1] = referenceBuilder.measure(r12[0]);
  auto [r2S2, cr2] = referenceBuilder.measure(r12[1]);
  referenceBuilder.sink(r0S1);
  referenceBuilder.sink(r1S2);
  referenceBuilder.sink(r2S2);

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
  auto q0S0 = programBuilder.allocQubit();
  auto q1S0 = programBuilder.allocQubit();
  auto q2S0 = programBuilder.allocQubit();

  auto [q0S1, c0] = programBuilder.measure(q0S0);
  auto [q1S1, c1] = programBuilder.measure(q1S0);

  auto [q01S1, q2S1] =
      programBuilder.ctrl({q0S1, q1S1}, {q2S0},
                          [&](const ValueRange targets) -> SmallVector<Value> {
                            auto q = programBuilder.x(targets[0]);
                            return SmallVector<Value>{q};
                          });

  auto [q2S2, c2] = programBuilder.measure(q2S1[0]);

  programBuilder.sink(q01S1[0]);
  programBuilder.sink(q01S1[1]);
  programBuilder.sink(q2S2);
  module = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r0S0 = referenceBuilder.allocQubit();
  auto r1S0 = referenceBuilder.allocQubit();
  auto r2S0 = referenceBuilder.allocQubit();

  auto [r0S1, cr0] = referenceBuilder.measure(r0S0);
  auto [r1S1, cr1] = referenceBuilder.measure(r1S0);

  auto andOp = arith::AndIOp::create(referenceBuilder, cr0, cr1);

  auto r2S1 = referenceBuilder.qcoIf(
      andOp.getResult(), {r2S0}, [&](ValueRange qubits) -> SmallVector<Value> {
        auto r = referenceBuilder.x(qubits[0]);
        return SmallVector<Value>{r};
      })[0];
  auto [r2S2, cr2] = referenceBuilder.measure(r2S1);
  referenceBuilder.sink(r0S1);
  referenceBuilder.sink(r1S1);
  referenceBuilder.sink(r2S2);

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
  auto q0S0 = programBuilder.allocQubit();
  auto q1S0 = programBuilder.allocQubit();
  auto q2S0 = programBuilder.allocQubit();
  auto q3S0 = programBuilder.allocQubit();

  auto [q0S1, c0] = programBuilder.measure(q0S0);
  auto [q1S1, c1] = programBuilder.measure(q1S0);

  auto [q012S1, q3S1] =
      programBuilder.ctrl({q0S1, q1S1, q2S0}, {q3S0},
                          [&](const ValueRange targets) -> SmallVector<Value> {
                            auto q = programBuilder.x(targets[0]);
                            return SmallVector<Value>{q};
                          });

  auto [q2S2, c2] = programBuilder.measure(q012S1[2]);
  auto [q3S2, c3] = programBuilder.measure(q3S1[0]);

  programBuilder.sink(q012S1[0]);
  programBuilder.sink(q012S1[1]);
  programBuilder.sink(q2S2);
  programBuilder.sink(q3S2);
  module = programBuilder.finalize({c0, c1, c2, c3});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type(),
       referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0S0 = referenceBuilder.allocQubit();
  auto r1S0 = referenceBuilder.allocQubit();
  auto r2S0 = referenceBuilder.allocQubit();
  auto r3S0 = referenceBuilder.allocQubit();

  auto [r0S1, cr0] = referenceBuilder.measure(r0S0);
  auto [r1S1, cr1] = referenceBuilder.measure(r1S0);

  auto andOp = arith::AndIOp::create(referenceBuilder, cr0, cr1);

  auto r23S1 =
      referenceBuilder.qcoIf(andOp.getResult(), {r2S0, r3S0},
                             [&](ValueRange qubits) -> SmallVector<Value> {
                               auto [r2, r3] =
                                   referenceBuilder.cx(qubits[0], qubits[1]);
                               return SmallVector<Value>{r2, r3};
                             });
  auto [r2S2, cr2] = referenceBuilder.measure(r23S1[0]);
  auto [r3S2, cr3] = referenceBuilder.measure(r23S1[1]);
  referenceBuilder.sink(r0S1);
  referenceBuilder.sink(r1S1);
  referenceBuilder.sink(r2S2);
  referenceBuilder.sink(r3S2);

  reference = referenceBuilder.finalize({cr0, cr1, cr2, cr3});

  ASSERT_TRUE(runReplaceClassicalControlsPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOReplaceClassicalControlsTest, replaceClassicalControlsSwapDiagonal) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0S0 = programBuilder.allocQubit();
  auto q1S0 = programBuilder.allocQubit();

  auto [q0S1, c0] = programBuilder.measure(q0S0);
  auto [q1S1, q0S2] = programBuilder.cz(q1S0, q0S1);
  auto [q1S2, c1] = programBuilder.measure(q1S1);

  programBuilder.sink(q0S2);
  programBuilder.sink(q1S2);
  module = programBuilder.finalize({c0, c1});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0S0 = referenceBuilder.allocQubit();
  auto r1S0 = referenceBuilder.allocQubit();

  auto [r0S1, cr0] = referenceBuilder.measure(r0S0);

  auto r1S1 = referenceBuilder.qcoIf(
      cr0, {r1S0}, [&](ValueRange qubits) -> SmallVector<Value> {
        auto q1Then = referenceBuilder.z(qubits[0]);
        return SmallVector<Value>{q1Then};
      })[0];
  auto [r1S2, cr1] = referenceBuilder.measure(r1S1);
  referenceBuilder.sink(r0S1);
  referenceBuilder.sink(r1S2);

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
  auto q0S0 = programBuilder.allocQubit();
  auto q1S0 = programBuilder.allocQubit();

  auto [q0S1, c0] = programBuilder.measure(q0S0);
  auto [q1S1, c1] = programBuilder.measure(q1S0);
  auto [q1S2, q0S2] = programBuilder.cz(q1S1, q0S1);
  auto [q0S3, c0_] = programBuilder.measure(q0S2);

  programBuilder.sink(q0S3);
  programBuilder.sink(q1S2);
  module = programBuilder.finalize({c0, c1, c0_});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               programBuilder.getI1Type()});
  auto r0S0 = referenceBuilder.allocQubit();
  auto r1S0 = referenceBuilder.allocQubit();

  auto [r0S1, cr0] = referenceBuilder.measure(r0S0);
  auto [r1S1, cr1] = referenceBuilder.measure(r1S0);

  auto r0S2 = referenceBuilder.qcoIf(
      cr1, {r0S1}, [&](ValueRange qubits) -> SmallVector<Value> {
        auto r0Then = referenceBuilder.z(qubits[0]);
        return SmallVector<Value>{r0Then};
      })[0];
  auto [r0S3, cr0_] = referenceBuilder.measure(r0S2);
  referenceBuilder.sink(r0S3);
  referenceBuilder.sink(r1S1);

  reference = referenceBuilder.finalize({cr0, cr1, cr0_});

  ASSERT_TRUE(runReplaceClassicalControlsPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  module->dump();

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

TEST_F(QCOReplaceClassicalControlsTest,
       replaceClassicalControlsSwapOneOfTwoDiagonal) {
  programBuilder.initialize(
      {programBuilder.getI1Type(), programBuilder.getI1Type()});
  auto q0S0 = programBuilder.allocQubit();
  auto q1S0 = programBuilder.allocQubit();
  auto q2S0 = programBuilder.allocQubit();

  auto [q0S1, c0] = programBuilder.measure(q0S0);
  auto [q12, q0S2] =
      programBuilder.ctrl({q1S0, q2S0}, {q0S1},
                          [&](const ValueRange targets) -> SmallVector<Value> {
                            auto q = programBuilder.z(targets[0]);
                            return SmallVector<Value>{q};
                          });
  auto [q1S2, c1] = programBuilder.measure(q12[0]);

  programBuilder.sink(q0S2[0]);
  programBuilder.sink(q1S2);
  programBuilder.sink(q12[1]);
  module = programBuilder.finalize({c0, c1});

  referenceBuilder.initialize(
      {referenceBuilder.getI1Type(), referenceBuilder.getI1Type()});
  auto r0S0 = referenceBuilder.allocQubit();
  auto r1S0 = referenceBuilder.allocQubit();
  auto r2S0 = referenceBuilder.allocQubit();

  auto [r0S1, cr0] = referenceBuilder.measure(r0S0);

  auto r21 = referenceBuilder.qcoIf(
      cr0, {r2S0, r1S0}, [&](ValueRange qubits) -> SmallVector<Value> {
        auto [r2, r1] = referenceBuilder.cz(qubits[0], qubits[1]);
        return SmallVector<Value>{r2, r1};
      });
  auto [r1S2, cr1] = referenceBuilder.measure(r21[1]);
  referenceBuilder.sink(r0S1);
  referenceBuilder.sink(r1S2);
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
  auto q0S0 = programBuilder.allocQubit();
  auto q1S0 = programBuilder.allocQubit();
  auto q2S0 = programBuilder.allocQubit();

  auto [q0S1, c0] = programBuilder.measure(q0S0);
  auto [q1S1, c1] = programBuilder.measure(q1S0);
  auto [q12, q0S2] =
      programBuilder.ctrl({q1S1, q2S0}, {q0S1},
                          [&](const ValueRange targets) -> SmallVector<Value> {
                            auto q = programBuilder.z(targets[0]);
                            return SmallVector<Value>{q};
                          });
  auto [q2S2, c2] = programBuilder.measure(q12[1]);

  programBuilder.sink(q0S2[0]);
  programBuilder.sink(q12[0]);
  programBuilder.sink(q2S2);
  module = programBuilder.finalize({c0, c1, c2});

  referenceBuilder.initialize({referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type(),
                               referenceBuilder.getI1Type()});
  auto r0S0 = referenceBuilder.allocQubit();
  auto r1S0 = referenceBuilder.allocQubit();
  auto r2S0 = referenceBuilder.allocQubit();

  auto [r0S1, cr0] = referenceBuilder.measure(r0S0);
  auto [r1S1, cr1] = referenceBuilder.measure(r1S0);

  auto andOp = arith::AndIOp::create(referenceBuilder, cr1, cr0);

  auto r2S1 = referenceBuilder.qcoIf(
      andOp.getResult(), {r2S0}, [&](ValueRange qubits) -> SmallVector<Value> {
        auto r = referenceBuilder.z(qubits[0]);
        return SmallVector<Value>{r};
      })[0];
  auto [r2S2, cr2] = referenceBuilder.measure(r2S1);
  referenceBuilder.sink(r0S1);
  referenceBuilder.sink(r1S1);
  referenceBuilder.sink(r2S2);

  reference = referenceBuilder.finalize({cr0, cr1, cr2});

  ASSERT_TRUE(runReplaceClassicalControlsPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}
