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
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

namespace {

using namespace mlir;
using namespace mlir::qco;

class QCOHadamardLiftingTest : public testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder programBuilder;
  QCOProgramBuilder referenceBuilder;
  OwningOpRef<ModuleOp> module;
  OwningOpRef<ModuleOp> reference;

  QCOHadamardLiftingTest()
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
  static LogicalResult runHadamardLiftingPass(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(createHadamardLifting());
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

// ##################################################
// # Raise Hadamard over uncontrolled Pauli gate Tests
// ##################################################

/**
 * @brief Test: Hadamards should be lifted over one Pauli gate. A global phase
 * should be added for the Pauli-Y gate.
 */
TEST_F(QCOHadamardLiftingTest, liftHadamardOverPauliGate) {
  auto q = programBuilder.allocQubitRegister(3);
  q[0] = programBuilder.x(q[0]);
  q[0] = programBuilder.h(q[0]);
  q[1] = programBuilder.z(q[1]);
  q[1] = programBuilder.h(q[1]);
  q[2] = programBuilder.y(q[2]);
  q[2] = programBuilder.h(q[2]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(3);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[0] = referenceBuilder.z(qRef[0]);
  qRef[1] = referenceBuilder.h(qRef[1]);
  qRef[1] = referenceBuilder.x(qRef[1]);
  qRef[2] = referenceBuilder.h(qRef[2]);
  qRef[2] = referenceBuilder.y(qRef[2]);
  referenceBuilder.gphase(M_PI);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Pauli gates should not be lifted over Hadamards.
 */
TEST_F(QCOHadamardLiftingTest, doNotLiftPauliOverHadamardGate) {
  auto q = programBuilder.allocQubitRegister(3);
  q[0] = programBuilder.h(q[0]);
  q[0] = programBuilder.x(q[0]);
  q[1] = programBuilder.h(q[1]);
  q[1] = programBuilder.z(q[1]);
  q[2] = programBuilder.h(q[2]);
  q[2] = programBuilder.y(q[2]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(3);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[0] = referenceBuilder.x(qRef[0]);
  qRef[1] = referenceBuilder.h(qRef[1]);
  qRef[1] = referenceBuilder.z(qRef[1]);
  qRef[2] = referenceBuilder.h(qRef[2]);
  qRef[2] = referenceBuilder.y(qRef[2]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Checks if Hadamard gates can be lifted over multiple Pauli gate.
 */
TEST_F(QCOHadamardLiftingTest, liftHadamardOverMultiplePauliGate) {
  auto q = programBuilder.allocQubitRegister(3);
  q[0] = programBuilder.x(q[0]);
  q[0] = programBuilder.z(q[0]);
  q[0] = programBuilder.h(q[0]);
  q[1] = programBuilder.x(q[1]);
  q[1] = programBuilder.y(q[1]);
  q[1] = programBuilder.z(q[1]);
  q[1] = programBuilder.h(q[1]);
  q[2] = programBuilder.x(q[2]);
  q[2] = programBuilder.s(q[2]);
  q[2] = programBuilder.x(q[2]);
  q[2] = programBuilder.y(q[2]);
  q[2] = programBuilder.h(q[2]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(3);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[0] = referenceBuilder.z(qRef[0]);
  qRef[0] = referenceBuilder.x(qRef[0]);
  qRef[1] = referenceBuilder.h(qRef[1]);
  qRef[1] = referenceBuilder.z(qRef[1]);
  qRef[1] = referenceBuilder.y(qRef[1]);
  referenceBuilder.gphase(M_PI);
  qRef[1] = referenceBuilder.x(qRef[1]);
  qRef[2] = referenceBuilder.x(qRef[2]);
  qRef[2] = referenceBuilder.s(qRef[2]);
  qRef[2] = referenceBuilder.h(qRef[2]);
  qRef[2] = referenceBuilder.z(qRef[2]);
  qRef[2] = referenceBuilder.y(qRef[2]);
  referenceBuilder.gphase(M_PI);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Checks if Hadamard gates are lifted over preceding and not over
 * succeeding Pauli gates.
 */
TEST_F(QCOHadamardLiftingTest, liftHadamardOnlyOverPrecedingPauliGate) {
  auto q = programBuilder.allocQubitRegister(2);
  q[0] = programBuilder.x(q[0]);
  q[0] = programBuilder.h(q[0]);
  q[0] = programBuilder.x(q[0]);
  q[1] = programBuilder.x(q[1]);
  q[1] = programBuilder.z(q[1]);
  q[1] = programBuilder.h(q[1]);
  q[1] = programBuilder.z(q[1]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[0] = referenceBuilder.z(qRef[0]);
  qRef[0] = referenceBuilder.x(qRef[0]);
  qRef[1] = referenceBuilder.h(qRef[1]);
  qRef[1] = referenceBuilder.z(qRef[1]);
  qRef[1] = referenceBuilder.x(qRef[1]);
  qRef[1] = referenceBuilder.z(qRef[1]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

// ##################################################
// # Raise Hadamard over controlled Pauli gate Tests
// ##################################################

/**
 * @brief Test: Checks if Hadamard gates are not lifted if they are controlled
 * by the same qubit as the lifted gate is.
 */
TEST_F(QCOHadamardLiftingTest, doNotLiftHadamardOverPauliGateIfControlled) {
  auto q = programBuilder.allocQubitRegister(2);
  q[0] = programBuilder.x(q[0]);
  auto qubitPair = programBuilder.cx(q[1], q[0]);
  qubitPair = programBuilder.ch(qubitPair.first, qubitPair.second);
  programBuilder.cx(qubitPair.first, qubitPair.second);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  qRef[0] = referenceBuilder.x(qRef[0]);
  auto qubitPairRef = referenceBuilder.cx(qRef[1], qRef[0]);
  qubitPairRef = referenceBuilder.ch(qubitPairRef.first, qubitPairRef.second);
  referenceBuilder.cx(qubitPairRef.first, qubitPairRef.second);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Checks that Hadamard gates are not lifted if they are controlled
 * and the Pauli gate is a Pauli-Y gate.
 */
TEST_F(QCOHadamardLiftingTest, doNotLiftHadamardOverPauliYGateIfControlled) {
  auto q = programBuilder.allocQubitRegister(2);
  q[0] = programBuilder.y(q[0]);
  auto qubitPair = programBuilder.cy(q[1], q[0]);
  qubitPair = programBuilder.ch(qubitPair.first, qubitPair.second);
  programBuilder.cy(qubitPair.first, qubitPair.second);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  qRef[0] = referenceBuilder.y(qRef[0]);
  auto qubitPairRef = referenceBuilder.cy(qRef[1], qRef[0]);
  qubitPairRef = referenceBuilder.ch(qubitPairRef.first, qubitPairRef.second);
  referenceBuilder.cy(qubitPairRef.first, qubitPairRef.second);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Checks that a hadamard gate is not lifted if they are controlled
 * by a different qubit than the one lifted gate is.
 */
TEST_F(QCOHadamardLiftingTest, doNotLiftHadamardIfDifferentControls) {
  auto q = programBuilder.allocQubitRegister(3);
  auto qubitPair = programBuilder.cx(q[1], q[0]);
  qubitPair = programBuilder.ch(q[2], qubitPair.second);
  q[0] = programBuilder.z(qubitPair.second);
  programBuilder.ch(qubitPair.first, q[0]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(3);
  auto qubitPairRef = referenceBuilder.cx(qRef[1], qRef[0]);
  qubitPairRef = referenceBuilder.ch(qRef[2], qubitPairRef.second);
  qRef[0] = referenceBuilder.z(qubitPairRef.second);
  referenceBuilder.ch(qubitPairRef.first, qRef[0]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Checks that a Hadamard gate is not lifted if there is another
 * gate between the controls of the Pauli and the Hadamard gate.
 */
TEST_F(QCOHadamardLiftingTest, doNotLiftHadamardIfGateBetweenControls) {
  auto q = programBuilder.allocQubitRegister(2);
  auto [q1, q2] = programBuilder.cz(q[1], q[0]);
  q[1] = programBuilder.s(q1);
  programBuilder.ch(q[1], q2);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  auto [q1Ref, q2Ref] = referenceBuilder.cz(qRef[1], qRef[0]);
  qRef[1] = referenceBuilder.s(q1Ref);
  referenceBuilder.ch(qRef[1], q2Ref);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Checks that a hadamard gate is not lifted if they do not share
 * all controls with the Pauli gate.
 */
TEST_F(QCOHadamardLiftingTest, doNotLiftHadamardIfSomeDifferentControls) {
  auto q = programBuilder.allocQubitRegister(3);
  auto [q12, q0] =
      programBuilder.ctrl({q[1], q[2]}, {q[0]}, [&](const ValueRange target) {
        return SmallVector{programBuilder.z(target[0])};
      });
  programBuilder.ch(q12[0], q0[0]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(3);
  auto [q12Ref, q0Ref] = referenceBuilder.ctrl(
      {qRef[1], qRef[2]}, {qRef[0]}, [&](const ValueRange target) {
        return SmallVector{referenceBuilder.z(target[0])};
      });
  referenceBuilder.ch(q12Ref[0], q0Ref[0]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

// ##################################################
// # Raise Hadamard over CNOT gates Tests
// ##################################################

/**
 * @brief Test: Checks that a Hadamard gate is lifted over a CNOT gate target if
 * a measurement is following directly after it.
 */
TEST_F(QCOHadamardLiftingTest, liftHadamardOverCNOTGate) {
  auto q = programBuilder.allocQubitRegister(2);
  const auto b = programBuilder.allocClassicalBitRegister(1);
  q[0] = programBuilder.s(q[0]);
  auto [q0, q1] = programBuilder.cx(q[0], q[1]);
  q[1] = programBuilder.h(q1);
  programBuilder.measure(q[1], b[0]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  const auto bRef = referenceBuilder.allocClassicalBitRegister(1);
  qRef[0] = referenceBuilder.s(qRef[0]);
  qRef[1] = referenceBuilder.h(qRef[1]);
  qRef[0] = referenceBuilder.h(qRef[0]);
  auto [q1Ref, q0Ref] = referenceBuilder.cx(qRef[1], qRef[0]);
  referenceBuilder.h(q0Ref);
  referenceBuilder.measure(q1Ref, bRef[0]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Checks that a Hadamard gate is lifted over the target of a
 * multiple controlled x gate if a measurement is following directly after it.
 */
TEST_F(QCOHadamardLiftingTest, liftHadamardOverMultipleControlledXGate) {
  auto q = programBuilder.allocQubitRegister(3);
  const auto b = programBuilder.allocClassicalBitRegister(1);
  auto [q12, q0] =
      programBuilder.ctrl({q[1], q[2]}, {q[0]}, [&](const ValueRange target) {
        return SmallVector{programBuilder.x(target[0])};
      });
  q[1] = programBuilder.h(q0[0]);
  programBuilder.measure(q[1], b[0]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(3);
  const auto bRef = referenceBuilder.allocClassicalBitRegister(1);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[1] = referenceBuilder.h(qRef[1]);
  auto [q02Ref, q1Ref] = referenceBuilder.ctrl(
      {qRef[0], qRef[2]}, {qRef[1]}, [&](const ValueRange target) {
        return SmallVector{referenceBuilder.x(target[0])};
      });
  referenceBuilder.h(q1Ref[0]);
  referenceBuilder.measure(q02Ref[0], bRef[0]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Checks that a Hadamard gate is not lifted over a CNOT gate
 * target if a measurement is not following directly after it.
 */
TEST_F(QCOHadamardLiftingTest, doNotLiftHadamardOverCNOTGate) {
  auto q = programBuilder.allocQubitRegister(6);
  const auto b = programBuilder.allocClassicalBitRegister(3);
  programBuilder.cx(q[1], q[0]);
  auto [q3, q2] = programBuilder.cx(q[3], q[2]);
  programBuilder.measure(q3, b[0]);
  auto [q5, q4] = programBuilder.cx(q[5], q[4]);
  q[4] = programBuilder.h(q4);
  q[5] = programBuilder.h(q5);
  q[4] = programBuilder.s(q[4]);
  programBuilder.measure(q[4], b[1]);
  programBuilder.measure(q[5], b[2]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(6);
  const auto bRef = referenceBuilder.allocClassicalBitRegister(3);
  referenceBuilder.cx(qRef[1], qRef[0]);
  auto [q3Ref, q2Ref] = referenceBuilder.cx(qRef[3], qRef[2]);
  referenceBuilder.measure(q3Ref, bRef[0]);
  auto [q5Ref, q4Ref] = referenceBuilder.cx(qRef[5], qRef[4]);
  qRef[4] = referenceBuilder.h(q4Ref);
  qRef[5] = referenceBuilder.h(q5Ref);
  qRef[4] = referenceBuilder.s(qRef[4]);
  referenceBuilder.measure(qRef[4], bRef[1]);
  referenceBuilder.measure(qRef[5], bRef[2]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Checks that a Hadamard gate is not lifted over a CNOT gate
 * target if a measurement is following directly after the controls.
 */
TEST_F(QCOHadamardLiftingTest,
       doNotLiftHadamardOverCNOTIfMeasurementsAfterControlsGate) {
  auto q = programBuilder.allocQubitRegister(5);
  const auto b = programBuilder.allocClassicalBitRegister(4);
  auto [q1, q0] = programBuilder.cx(q[1], q[0]);
  q[0] = programBuilder.h(q0);
  programBuilder.measure(q[0], b[0]);
  programBuilder.measure(q1, b[1]);
  auto [q34, q2] =
      programBuilder.ctrl({q[3], q[4]}, {q[2]}, [&](const ValueRange target) {
        return SmallVector{programBuilder.x(target[0])};
      });
  q[2] = programBuilder.h(q2[0]);
  programBuilder.measure(q[2], b[2]);
  programBuilder.measure(q34[0], b[3]);
  programBuilder.s(q34[1]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(5);
  const auto bRef = referenceBuilder.allocClassicalBitRegister(4);
  auto [qRef1, qRef0] = referenceBuilder.cx(qRef[1], qRef[0]);
  qRef[0] = referenceBuilder.h(qRef0);
  referenceBuilder.measure(qRef[0], bRef[0]);
  referenceBuilder.measure(qRef1, bRef[1]);
  qRef[2] = referenceBuilder.h(qRef[2]);
  qRef[4] = referenceBuilder.h(qRef[4]);
  auto [qRef32, qRef4] = referenceBuilder.ctrl(
      {qRef[3], qRef[2]}, {qRef[4]}, [&](const ValueRange target) {
        return SmallVector{referenceBuilder.x(target[0])};
      });
  qRef[4] = referenceBuilder.h(qRef4[0]);
  referenceBuilder.measure(qRef32[1], bRef[2]);
  referenceBuilder.measure(qRef32[0], bRef[3]);
  referenceBuilder.s(qRef[4]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}
