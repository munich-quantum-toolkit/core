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
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>

#include <cassert>
#include <cmath>

namespace {

using namespace mlir;
using namespace mlir::qco;

class QCOHadamardLiftingTest : public ::testing::Test {
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
};

} // namespace

// ##################################################
// # Raise Hadamard over uncontrolled Pauli gate Tests
// ##################################################

/**
 * @brief Test: Hadamards should be lifted over one Pauli gate.
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
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  runCanonicalizationPasses(reference.get());
  EXPECT_TRUE(verify(*reference).succeeded());

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
  runCanonicalizationPasses(reference.get());
  EXPECT_TRUE(verify(*reference).succeeded());

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
  qRef[1] = referenceBuilder.x(qRef[1]);
  qRef[2] = referenceBuilder.x(qRef[2]);
  qRef[2] = referenceBuilder.s(qRef[2]);
  qRef[2] = referenceBuilder.h(qRef[2]);
  qRef[2] = referenceBuilder.z(qRef[2]);
  qRef[2] = referenceBuilder.y(qRef[2]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  runCanonicalizationPasses(reference.get());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Checks if Hadamard gates are lifted over preceding and not over
 * succeeding Pauli gates.
 */
TEST_F(QCOHadamardLiftingTest, liftHadamardOnlyOverPreceedingPauliGate) {
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
  runCanonicalizationPasses(reference.get());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

// ##################################################
// # Raise Hadamard over controlled Pauli gate Tests
// ##################################################

/**
 * @brief Test: Checks if hadamard gates are lifted if they are controlled by
 * the same qubit as the lifted gate is.
 */
TEST_F(QCOHadamardLiftingTest, liftHadamardOverPauliGateIfControlled) {
  auto q = programBuilder.allocQubitRegister(2);
  q[0] = programBuilder.x(q[0]);
  auto qubit_pair = programBuilder.dcx(q[1], q[0]);
  auto qubitPairRange = programBuilder.ctrl(
      {qubit_pair.first}, {qubit_pair.second}, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{programBuilder.h(target[0])};
      });
  programBuilder.dcx(qubitPairRange.first[0], qubitPairRange.second[0]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  qRef[0] = referenceBuilder.x(qRef[0]);
  auto qubitPairRangeRef =
      referenceBuilder.ctrl({qRef[1]}, {qRef[0]}, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{referenceBuilder.h(target[0])};
      });
  auto qubitPairRef = referenceBuilder.cz(qubitPairRangeRef.first[0],
                                          qubitPairRangeRef.second[0]);
  referenceBuilder.dcx(qubitPairRef.first, qubitPairRef.second);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  runCanonicalizationPasses(reference.get());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Checks that a hadamard gate is not lifted if they are controlled
 * by a different qubit than the one lifted gate is.
 */
TEST_F(QCOHadamardLiftingTest, doNotLiftHadamardIfDifferentControls) {
  auto q = programBuilder.allocQubitRegister(3);
  auto qubit_pair = programBuilder.dcx(q[1], q[0]);
  auto qubitPairRange = programBuilder.ctrl(
      {q[2]}, {qubit_pair.second}, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{programBuilder.h(target[0])};
      });
  q[0] = programBuilder.z(qubitPairRange.second[0]);
  programBuilder.ctrl({qubit_pair.first}, {q[0]}, [&](mlir::ValueRange target) {
    return llvm::SmallVector<mlir::Value>{programBuilder.h(target[0])};
  });
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(3);
  auto qubitPairRef = referenceBuilder.dcx(qRef[1], qRef[0]);
  auto qubitPairRangeRef = referenceBuilder.ctrl(
      {qRef[2]}, {qubitPairRef.second}, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{referenceBuilder.h(target[0])};
      });
  qRef[0] = referenceBuilder.z(qubitPairRangeRef.second[0]);
  referenceBuilder.ctrl(
      {qubitPairRef.first}, {qRef[0]}, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{referenceBuilder.h(target[0])};
      });
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  runCanonicalizationPasses(reference.get());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Checks that a Hadamard gate is not lifted if there is another
 * gate between the controls of the Pauli and the Hadamard gate.
 */
TEST_F(QCOHadamardLiftingTest, doNotLiftHadamardIfGateBetweenControls) {
  auto q = programBuilder.allocQubitRegister(2);
  auto qubitPairRange =
      programBuilder.ctrl({q[1]}, {q[0]}, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{programBuilder.z(target[0])};
      });
  q[1] = programBuilder.s(qubitPairRange.first[0]);
  programBuilder.ctrl(
      {q[1]}, qubitPairRange.second, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{programBuilder.h(target[0])};
      });
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  auto qubitPairRangeRef =
      referenceBuilder.ctrl({qRef[1]}, {qRef[0]}, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{referenceBuilder.z(target[0])};
      });
  qRef[1] = referenceBuilder.s(qubitPairRangeRef.first[0]);
  referenceBuilder.ctrl(
      {qRef[1]}, qubitPairRangeRef.second[0], [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{referenceBuilder.h(target[0])};
      });
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  runCanonicalizationPasses(reference.get());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Checks that a hadamard gate is not lifted if they do not share
 * all controls with the Pauli gate.
 */
TEST_F(QCOHadamardLiftingTest, doNotLiftHadamardIfSomeDifferentControls) {
  auto q = programBuilder.allocQubitRegister(3);
  auto qubitPairRange =
      programBuilder.ctrl({q[1], q[2]}, {q[0]}, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{programBuilder.z(target[0])};
      });
  programBuilder.ctrl({qubitPairRange.first[0]}, qubitPairRange.second,
                      [&](mlir::ValueRange target) {
                        return llvm::SmallVector<mlir::Value>{
                            programBuilder.h(target[0])};
                      });
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(3);
  auto qubitPairRangeRef = referenceBuilder.ctrl(
      {qRef[1], qRef[2]}, {qRef[0]}, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{referenceBuilder.z(target[0])};
      });
  referenceBuilder.ctrl({qubitPairRangeRef.first[0]}, qubitPairRangeRef.second,
                        [&](mlir::ValueRange target) {
                          return llvm::SmallVector<mlir::Value>{
                              referenceBuilder.h(target[0])};
                        });
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  runCanonicalizationPasses(reference.get());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Checks that a hadamard gate can be lifted over a controlled
 * Pauli Z gate even if the targets are at different places.
 */
TEST_F(QCOHadamardLiftingTest, liftHadamardOverControlledPauliZ) {
  auto q = programBuilder.allocQubitRegister(3);
  auto qubitPairRange =
      programBuilder.ctrl({q[1], q[2]}, {q[0]}, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{programBuilder.z(target[0])};
      });
  qubitPairRange = programBuilder.ctrl(
      {qubitPairRange.first[1], qubitPairRange.second[0]},
      {qubitPairRange.first[0]}, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{programBuilder.h(target[0])};
      });
  auto qubitPairRangeOne = programBuilder.ctrl(
      qubitPairRange.second, {qubitPairRange.first[0]},
      [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{programBuilder.z(target[0])};
      });
  qubitPairRange = programBuilder.ctrl(
      {qubitPairRange.first[1], qubitPairRangeOne.second[0]},
      qubitPairRangeOne.first, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{programBuilder.h(target[0])};
      });
  qubitPairRange = programBuilder.ctrl(
      qubitPairRange.first, qubitPairRange.second,
      [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{programBuilder.z(target[0])};
      });
  programBuilder.ctrl(qubitPairRange.second, {qubitPairRange.first[0]},
                      [&](mlir::ValueRange target) {
                        return llvm::SmallVector<mlir::Value>{
                            programBuilder.z(target[0])};
                      });
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(3);
  auto qubitPairRangeRef = referenceBuilder.ctrl(
      {qRef[2], qRef[0]}, {qRef[1]}, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{referenceBuilder.h(target[0])};
      });
  qubitPairRangeRef = referenceBuilder.ctrl(
      qubitPairRangeRef.first, qubitPairRangeRef.second,
      [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{referenceBuilder.x(target[0])};
      });
  auto qubitPairRangeOneRef = referenceBuilder.ctrl(
      qubitPairRangeRef.second, {qubitPairRangeRef.first[0]},
      [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{referenceBuilder.z(target[0])};
      });
  qubitPairRangeRef = referenceBuilder.ctrl(
      {qubitPairRangeRef.first[1], qubitPairRangeOneRef.second[0]},
      qubitPairRangeOneRef.first, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{referenceBuilder.h(target[0])};
      });
  qubitPairRangeRef = referenceBuilder.ctrl(
      qubitPairRangeRef.first, qubitPairRangeRef.second,
      [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{referenceBuilder.z(target[0])};
      });
  referenceBuilder.ctrl(qubitPairRangeRef.second, {qubitPairRangeRef.first[0]},
                        [&](mlir::ValueRange target) {
                          return llvm::SmallVector<mlir::Value>{
                              referenceBuilder.z(target[0])};
                        });
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  runCanonicalizationPasses(reference.get());
  EXPECT_TRUE(verify(*reference).succeeded());

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
  auto b = programBuilder.allocClassicalBitRegister(1);
  q[0] = programBuilder.s(q[0]);
  auto qubitPair = programBuilder.dcx(q[0], q[1]);
  q[1] = programBuilder.h(qubitPair.second);
  programBuilder.measure(q[1], b[0]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  auto bRef = referenceBuilder.allocClassicalBitRegister(1);
  qRef[0] = referenceBuilder.s(qRef[0]);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[1] = referenceBuilder.h(qRef[1]);
  auto qubitPairRef = referenceBuilder.dcx(qRef[1], qRef[0]);
  referenceBuilder.h(qubitPairRef.first);
  referenceBuilder.measure(qubitPairRef.second, bRef[0]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  runCanonicalizationPasses(reference.get());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Checks that a Hadamard gate is lifted over the target of a
 * multiple controlled x gate if a measurement is following directly after it.
 */
TEST_F(QCOHadamardLiftingTest, liftHadamardOverMultipleControlledXGate) {
  auto q = programBuilder.allocQubitRegister(3);
  auto b = programBuilder.allocClassicalBitRegister(1);
  auto qubitPairRange =
      programBuilder.ctrl({q[1], q[2]}, {q[0]}, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{programBuilder.x(target[0])};
      });
  q[1] = programBuilder.h(qubitPairRange.second[0]);
  programBuilder.measure(q[1], b[0]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(3);
  auto bRef = referenceBuilder.allocClassicalBitRegister(1);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[1] = referenceBuilder.h(qRef[1]);
  auto qubitPairRangeRef = referenceBuilder.ctrl(
      {qRef[1], qRef[2]}, {qRef[0]}, [&](mlir::ValueRange target) {
        return llvm::SmallVector<mlir::Value>{referenceBuilder.x(target[0])};
      });
  referenceBuilder.h(qubitPairRangeRef.first[0]);
  referenceBuilder.measure(qubitPairRangeRef.second[0], bRef[0]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  runCanonicalizationPasses(reference.get());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: Checks that a Hadamard gate is not lifted over a CNOT gate
 * target if a measurement is not following directly after it.
 */
TEST_F(QCOHadamardLiftingTest, doNotLiftHadamardOverCNOTGate) {
  auto q = programBuilder.allocQubitRegister(6);
  auto b = programBuilder.allocClassicalBitRegister(3);
  programBuilder.dcx(q[1], q[0]);
  auto qubitPairOne = programBuilder.dcx(q[3], q[2]);
  programBuilder.measure(qubitPairOne.first, b[0]);
  auto qubitPairTwo = programBuilder.dcx(q[5], q[4]);
  q[4] = programBuilder.h(qubitPairTwo.second);
  q[5] = programBuilder.h(qubitPairTwo.first);
  q[5] = programBuilder.s(q[5]);
  programBuilder.measure(q[4], b[1]);
  programBuilder.measure(q[5], b[2]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(6);
  auto bRef = referenceBuilder.allocClassicalBitRegister(3);
  referenceBuilder.dcx(qRef[1], qRef[0]);
  auto qubitPairOneRef = referenceBuilder.dcx(qRef[3], qRef[2]);
  referenceBuilder.measure(qubitPairOneRef.first, bRef[0]);
  auto qubitPairTwoRef = referenceBuilder.dcx(qRef[5], qRef[4]);
  qRef[4] = referenceBuilder.h(qubitPairTwoRef.second);
  qRef[5] = referenceBuilder.h(qubitPairTwoRef.first);
  qRef[5] = referenceBuilder.s(qRef[5]);
  referenceBuilder.measure(qRef[4], bRef[1]);
  referenceBuilder.measure(qRef[5], bRef[2]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runHadamardLiftingPass(module.get()).succeeded());
  runCanonicalizationPasses(reference.get());
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}
