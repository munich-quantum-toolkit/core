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
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <numbers>

namespace {

using namespace mlir;
using namespace mlir::qco;

class QCOConstantPropagationTest : public testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder programBuilder;
  QCOProgramBuilder referenceBuilder;
  OwningOpRef<ModuleOp> module;
  OwningOpRef<ModuleOp> reference;

  QCOConstantPropagationTest()
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
  static LogicalResult runConstantPropagationPass(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(createConstantPropagation());
    return pm.run(module);
  }
};

} // namespace

/**
 * @brief Test: This test checks if CNOTs or the controls of CNOTs are removed
 * if we can classically determine the ctrls value.
 */
TEST_F(QCOConstantPropagationTest, reducePosCtrls) {
  const auto iAttr = programBuilder.getF64FloatAttr(-0.3926991);
  Value i0 =
      arith::ConstantOp::create(programBuilder, programBuilder.getLoc(), iAttr);
  auto q = programBuilder.allocQubitRegister(4);
  q[0] = programBuilder.h(q[0]);
  q[0] = programBuilder.x(q[0]);
  q[0] = programBuilder.h(q[0]);
  programBuilder.crx(i0, q[0], q[1]);
  q[2] = programBuilder.h(q[2]);
  q[2] = programBuilder.z(q[2]);
  q[2] = programBuilder.h(q[2]);
  auto [q2, q3] = programBuilder.crx(i0, q[2], q[3]);
  programBuilder.cry(0.3, q2, q3);
  module = programBuilder.finalize();

  const auto iAttrRef = referenceBuilder.getF64FloatAttr(-0.3926991);
  Value i0Ref = arith::ConstantOp::create(referenceBuilder,
                                          referenceBuilder.getLoc(), iAttrRef);
  auto qRef = referenceBuilder.allocQubitRegister(4);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[0] = referenceBuilder.x(qRef[0]);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[2] = referenceBuilder.h(qRef[2]);
  qRef[2] = referenceBuilder.z(qRef[2]);
  qRef[2] = referenceBuilder.h(qRef[2]);
  qRef[3] = referenceBuilder.rx(i0Ref, qRef[3]);
  referenceBuilder.ry(0.3, qRef[3]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks that CNOTs are not changed if the target is not
 * in |0> or |1>.
 */
TEST_F(QCOConstantPropagationTest, testDontRemoveIfTargetInSuperposition) {
  auto q = programBuilder.allocQubitRegister(2);
  q[0] = programBuilder.h(q[0]);
  programBuilder.cx(q[0], q[1]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  qRef[0] = referenceBuilder.h(qRef[0]);
  referenceBuilder.cx(qRef[0], qRef[1]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks that CNOTs are not changed if a reset is
 * between to Hadamards, i.e. the qubits are in a superposition after the second
 * Hadamard.
 */
TEST_F(QCOConstantPropagationTest, testApplyReset) {
  auto q = programBuilder.allocQubitRegister(2);
  q[0] = programBuilder.h(q[0]);
  q[0] = programBuilder.reset(q[0]);
  q[0] = programBuilder.h(q[0]);
  programBuilder.cx(q[0], q[1]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[0] = referenceBuilder.reset(qRef[0]);
  qRef[0] = referenceBuilder.h(qRef[0]);
  referenceBuilder.cx(qRef[0], qRef[1]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks that implied Qubits are removed from a
 * controlled gate.
 */
TEST_F(QCOConstantPropagationTest, testRemoveImpliedQubits) {
  auto q = programBuilder.allocQubitRegister(5);
  const auto iAttr = programBuilder.getF64FloatAttr(-0.3926991);
  Value i0 =
      arith::ConstantOp::create(programBuilder, programBuilder.getLoc(), iAttr);
  q[0] = programBuilder.h(q[0]);
  q[1] = programBuilder.h(q[1]);
  auto [q01, q2] =
      programBuilder.ctrl({q[0], q[1]}, {q[2]}, [&](const ValueRange target) {
        return SmallVector{programBuilder.x(target[0])};
      });
  q[4] = programBuilder.x(q[4]);
  auto [q124, q3] = programBuilder.ctrl(
      {q01[1], q2[0], q[4]}, {q[3]}, [&](const ValueRange target) {
        return SmallVector{programBuilder.rx(i0, target[0])};
      });
  programBuilder.h(q01[0]);
  programBuilder.h(q124[0]);
  programBuilder.h(q124[1]);
  programBuilder.h(q3[0]);
  programBuilder.h(q124[2]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(5);
  const auto iAttrRef = referenceBuilder.getF64FloatAttr(-0.3926991);
  Value i0Ref = arith::ConstantOp::create(referenceBuilder,
                                          referenceBuilder.getLoc(), iAttrRef);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[1] = referenceBuilder.h(qRef[1]);
  auto [qRef01, qRef2] = referenceBuilder.ctrl(
      {qRef[0], qRef[1]}, {qRef[2]}, [&](const ValueRange target) {
        return SmallVector{referenceBuilder.x(target[0])};
      });
  qRef[4] = referenceBuilder.x(qRef[4]);
  auto [qRef21, qRef31] = referenceBuilder.crx(i0Ref, qRef2[0], qRef[3]);
  referenceBuilder.h(qRef01[0]);
  referenceBuilder.h(qRef01[1]);
  referenceBuilder.h(qRef21);
  referenceBuilder.h(qRef31);
  referenceBuilder.h(qRef[4]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks that gates whose quantum controls cannot be
 * satisfied are removed.
 */
TEST_F(QCOConstantPropagationTest, testUnsatisfiableQuantumCombination) {
  auto q = programBuilder.allocQubitRegister(3);
  q[0] = programBuilder.h(q[0]);
  q[1] = programBuilder.x(q[1]);
  auto [q0, q1] = programBuilder.cx(q[0], q[1]);
  programBuilder.ctrl({q0, q1}, {q[2]}, [&](const ValueRange target) {
    return SmallVector{programBuilder.s(target[0])};
  });
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(3);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[1] = referenceBuilder.x(qRef[1]);
  referenceBuilder.cx(qRef[0], qRef[1]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks that gates whose quantum and classical controls
 * cannot be satisfied are removed.
 */
TEST_F(QCOConstantPropagationTest, testUnsatisfiableHybridCombination) {
  auto q = programBuilder.allocQubitRegister(2);
  q[0] = programBuilder.h(q[0]);
  q[1] = programBuilder.x(q[1]);
  auto [q0, q1] = programBuilder.cx(q[0], q[1]);
  auto [q01, b0] = programBuilder.measure(q0);
  const auto qRange01 = programBuilder.qcoIf(
      b0, {q01, q1},
      [&](const ValueRange args) { return SmallVector{args[0], args[1]}; },
      [&](const ValueRange args) {
        const auto [qi0, qi1] = programBuilder.ch(args[0], args[1]);
        return SmallVector{qi0, qi1};
      });
  programBuilder.y(qRange01[1]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[1] = referenceBuilder.x(qRef[1]);
  auto [qRef0, qRef1] = referenceBuilder.cx(qRef[0], qRef[1]);
  referenceBuilder.measure(qRef0);
  referenceBuilder.y(qRef1);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks that gates are unconditionally applied if the
 * bit they depend on is always zero.
 */
TEST_F(QCOConstantPropagationTest, testRemoveClassicalConditionalIfItsZero) {
  auto q = programBuilder.allocQubitRegister(1);
  q[0] = programBuilder.h(q[0]);
  q[0] = programBuilder.h(q[0]);
  auto [q0, b0] = programBuilder.measure(q[0]);
  programBuilder.qcoIf(
      b0, {q0},
      [&](const ValueRange args) {
        const auto qi0 = programBuilder.x(args[0]);
        return SmallVector{qi0};
      },
      [&](const ValueRange args) {
        const auto qi0 = programBuilder.h(args[0]);
        return SmallVector{qi0};
      });
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(1);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[0] = referenceBuilder.h(qRef[0]);
  auto [qRef0, bRef0] = referenceBuilder.measure(qRef[0]);
  referenceBuilder.h(qRef0);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks that gates are unconditionally applied if the
 * bit they depend on is always one.
 */
TEST_F(QCOConstantPropagationTest, testRemoveClassicalConditionalIfItsOne) {
  auto q = programBuilder.allocQubitRegister(2);
  q[0] = programBuilder.x(q[0]);
  auto [q0, b0] = programBuilder.measure(q[0]);
  const auto qRange01 = programBuilder.qcoIf(
      b0, {q0, q[1]},
      [&](const ValueRange args) {
        const auto qi0 = programBuilder.h(args[0]);
        const auto qi1 = programBuilder.h(args[1]);
        const auto qi11 = programBuilder.z(qi1);
        const auto [qi2, qi3] = programBuilder.cx(qi0, qi11);
        return SmallVector{qi2, qi3};
      },
      [&](const ValueRange args) {
        const auto qi0 = programBuilder.h(args[0]);
        return SmallVector{qi0, args[1]};
      });
  programBuilder.h(qRange01[1]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  qRef[0] = referenceBuilder.x(qRef[0]);
  auto [qRef0, bRef0] = referenceBuilder.measure(qRef[0]);
  qRef[0] = referenceBuilder.h(qRef0);
  qRef[1] = referenceBuilder.h(qRef[1]);
  qRef[1] = referenceBuilder.z(qRef[1]);
  const auto [qRef01, qRef1] = referenceBuilder.cx(qRef[0], qRef[1]);
  referenceBuilder.h(qRef1);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks for various gates if they are removed if the
 * classical conditional is true.
 */
TEST_F(QCOConstantPropagationTest, testRemoveClassicalConditionals) {
  auto q = programBuilder.allocQubitRegister(1);
  const auto bTrue = arith::ConstantOp::create(programBuilder,
                                               programBuilder.getBoolAttr(true))
                         .getResult();
  const auto bFalse = arith::ConstantOp::create(
                          programBuilder, programBuilder.getBoolAttr(false))
                          .getResult();
  auto b = arith::OrIOp::create(programBuilder, bTrue.getType(), bTrue, bFalse)
               .getResult();
  q[0] = programBuilder.h(q[0]);
  programBuilder.qcoIf(b, {q[0]}, [&](const ValueRange args) {
    const auto qi0 = programBuilder.u2(1.4, 2.7, args[0]);
    const auto qi1 = programBuilder.sdg(qi0);
    const auto qi2 = programBuilder.t(qi1);
    const auto qi3 = programBuilder.sx(qi2);
    const auto qi4 = programBuilder.tdg(qi3);
    const auto qi5 = programBuilder.sxdg(qi4);
    const auto qi6 = programBuilder.r(0.2, 0.4, qi5);
    return SmallVector{qi6};
  });
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(1);
  const auto bTrueRef =
      arith::ConstantOp::create(referenceBuilder,
                                referenceBuilder.getBoolAttr(true))
          .getResult();
  const auto bFalseRef =
      arith::ConstantOp::create(referenceBuilder,
                                referenceBuilder.getBoolAttr(false))
          .getResult();
  arith::OrIOp::create(referenceBuilder, bTrue.getType(), bTrueRef, bFalseRef)
      .getResult();
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[0] = referenceBuilder.u2(1.4, 2.7, qRef[0]);
  qRef[0] = referenceBuilder.sdg(qRef[0]);
  qRef[0] = referenceBuilder.t(qRef[0]);
  qRef[0] = referenceBuilder.sx(qRef[0]);
  qRef[0] = referenceBuilder.tdg(qRef[0]);
  qRef[0] = referenceBuilder.sxdg(qRef[0]);
  qRef[0] = referenceBuilder.r(0.2, 0.4, qRef[0]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks that conditionals are not changed if we cannot
 * tell the bits value.
 */
TEST_F(QCOConstantPropagationTest, testDoNotRemoveClassicalConditional) {
  auto q = programBuilder.allocQubitRegister(1);
  q[0] = programBuilder.h(q[0]);
  auto [q0, b0] = programBuilder.measure(q[0]);
  programBuilder.qcoIf(
      b0, {q0},
      [&](const ValueRange args) {
        const auto qi0 = programBuilder.x(args[0]);
        return SmallVector{qi0};
      },
      [&](const ValueRange args) {
        const auto qi0 = programBuilder.h(args[0]);
        return SmallVector{qi0};
      });
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(1);
  qRef[0] = referenceBuilder.h(qRef[0]);
  auto [qRef0, bRef0] = referenceBuilder.measure(qRef[0]);
  referenceBuilder.qcoIf(
      bRef0, {qRef0},
      [&](const ValueRange args) {
        const auto qi0 = referenceBuilder.x(args[0]);
        return SmallVector{qi0};
      },
      [&](const ValueRange args) {
        const auto qi0 = referenceBuilder.h(args[0]);
        return SmallVector{qi0};
      });
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks that a quantum conditional is replaced by a
 * classical if a qubit and a classical bit are equivalent.
 */
TEST_F(QCOConstantPropagationTest,
       testEquivalentPositiveClassicalAndQuantumControl) {
  auto q = programBuilder.allocQubitRegister(3);
  q[0] = programBuilder.h(q[0]);
  auto [q0, q1] = programBuilder.cx(q[0], q[1]);
  programBuilder.measure(q0);
  auto [q11, q2] = programBuilder.cx(q1, q[2]);
  programBuilder.h(q11);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(3);
  qRef[0] = referenceBuilder.h(qRef[0]);
  auto [qRef0, qRef1] = referenceBuilder.cx(qRef[0], qRef[1]);
  auto [qRef01, bRef0] = referenceBuilder.measure(qRef0);
  referenceBuilder.qcoIf(bRef0, {qRef[2]}, [&](const ValueRange args) {
    const auto qi0 = referenceBuilder.x(args[0]);
    return SmallVector{qi0};
  });
  referenceBuilder.h(qRef1);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks that multiple quantum conditionals are replaced
 * by a classical if a qubit and a classical bit are equivalent.
 */
TEST_F(QCOConstantPropagationTest, testEquivalentClassicalAndQuantumControl) {
  auto q = programBuilder.allocQubitRegister(3);
  q[0] = programBuilder.h(q[0]);
  auto [q0, q1] = programBuilder.cx(q[0], q[1]);
  auto [q01, b0] = programBuilder.measure(q0);
  auto [q11, q2] = programBuilder.cx(q1, q[2]);
  q[1] = programBuilder.x(q11);
  auto [q12, q21] = programBuilder.cy(q[1], q2);
  programBuilder.x(q01);
  programBuilder.y(q12);
  programBuilder.h(q21);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(3);
  qRef[0] = referenceBuilder.h(qRef[0]);
  auto [qRef0, qRef1] = referenceBuilder.cx(qRef[0], qRef[1]);
  auto [qRef01, bRef0] = referenceBuilder.measure(qRef0);
  const auto qRange2 =
      referenceBuilder.qcoIf(bRef0, {qRef[2]}, [&](const ValueRange args) {
        const auto qi0 = referenceBuilder.x(args[0]);
        return SmallVector{qi0};
      });
  qRef[1] = referenceBuilder.x(qRef1);
  const auto qRange21 = referenceBuilder.qcoIf(
      bRef0, qRange2,
      [&](const ValueRange args) { return SmallVector{args[0]}; },
      [&](const ValueRange args) {
        const auto qi0 = referenceBuilder.y(args[0]);
        return SmallVector{qi0};
      });
  referenceBuilder.x(qRef01);
  referenceBuilder.y(qRef[1]);
  referenceBuilder.h(qRange21[0]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks if a quantum control is removed if the
 * classical control implies the quantum one.
 */
TEST_F(QCOConstantPropagationTest, testClassicalImpliesQuantum) {
  auto q = programBuilder.allocQubitRegister(2);
  q[0] = programBuilder.h(q[0]);
  auto [q0, b0] = programBuilder.measure(q[0]);
  q[1] = programBuilder.x(q[1]);
  auto [q01, q1] = programBuilder.cx(q0, q[1]);
  auto [q11, q02] = programBuilder.ch(q1, q01);
  const auto qRange =
      programBuilder.qcoIf(b0, {q02, q11}, [&](const ValueRange args) {
        const auto [qi0, qi1] = programBuilder.cx(args[0], args[1]);
        return SmallVector{qi0, qi1};
      });
  programBuilder.x(qRange[0]);
  programBuilder.y(qRange[1]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  qRef[0] = referenceBuilder.h(qRef[0]);
  auto [qRef0, bRef0] = referenceBuilder.measure(qRef[0]);
  qRef[1] = referenceBuilder.x(qRef[1]);
  const auto qRefRange1 =
      referenceBuilder.qcoIf(bRef0, {qRef[1]}, [&](const ValueRange args) {
        const auto qi0 = referenceBuilder.x(args[0]);
        return SmallVector{qi0};
      });
  const auto qRefRange0 = referenceBuilder.qcoIf(
      bRef0, {qRef0},
      [&](const ValueRange args) { return SmallVector{args[0]}; },
      [&](const ValueRange args) {
        const auto qi0 = referenceBuilder.h(args[0]);
        return SmallVector{qi0};
      });
  const auto qRefRange = referenceBuilder.qcoIf(
      bRef0, {qRefRange0[0], qRefRange1[0]}, [&](const ValueRange args) {
        const auto qi0 = referenceBuilder.x(args[1]);
        return SmallVector{args[0], qi0};
      });
  referenceBuilder.x(qRefRange[0]);
  referenceBuilder.y(qRefRange[1]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks if propagation through classical branching is
 * done correctly.
 */
TEST_F(QCOConstantPropagationTest, testPropagatingThroughClassicalBranching) {
  auto q = programBuilder.allocQubitRegister(2);
  q[0] = programBuilder.h(q[0]);
  q[1] = programBuilder.x(q[1]);
  auto [q0, q1] = programBuilder.cx(q[0], q[1]);
  auto [q01, b0] = programBuilder.measure(q0);
  const auto qRange = programBuilder.qcoIf(
      b0, {q01, q1},
      [&](const ValueRange args) {
        const auto qubit = programBuilder.x(args[1]);
        return SmallVector{args[0], qubit};
      },
      [&](const ValueRange args) {
        const auto qubit = programBuilder.x(args[0]);
        return SmallVector{qubit, args[1]};
      });
  programBuilder.cz(qRange[0], qRange[1]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[1] = referenceBuilder.x(qRef[1]);
  auto [qRef0, qRef1] = referenceBuilder.cx(qRef[0], qRef[1]);
  auto [qRef01, bRef0] = referenceBuilder.measure(qRef0);
  referenceBuilder.qcoIf(
      bRef0, {qRef01, qRef1},
      [&](const ValueRange args) {
        const auto qubit = referenceBuilder.x(args[1]);
        return SmallVector{args[0], qubit};
      },
      [&](const ValueRange args) {
        const auto qubit = referenceBuilder.x(args[0]);
        return SmallVector{qubit, args[1]};
      });
  referenceBuilder.gphase(std::numbers::pi);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks if a phase gate is removed if it only adds a
 * global phase = 1.
 */
TEST_F(QCOConstantPropagationTest, testReplaceSingleQubitPhaseGatePlusOne) {
  auto q = programBuilder.allocQubitRegister(1);
  q[0] = programBuilder.h(q[0]);
  q[0] = programBuilder.z(q[0]);
  q[0] = programBuilder.z(q[0]);
  q[0] = programBuilder.h(q[0]);
  q[0] = programBuilder.z(q[0]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(1);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[0] = referenceBuilder.z(qRef[0]);
  qRef[0] = referenceBuilder.z(qRef[0]);
  qRef[0] = referenceBuilder.h(qRef[0]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks if a phase gate is replaced by a global phase
 * gate if it only adds a global phase.
 */
TEST_F(QCOConstantPropagationTest, testReplaceSingleQubitPhaseGateMinusOne) {
  auto q = programBuilder.allocQubitRegister(1);
  q[0] = programBuilder.h(q[0]);
  q[0] = programBuilder.z(q[0]);
  q[0] = programBuilder.h(q[0]);
  q[0] = programBuilder.z(q[0]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(1);
  qRef[0] = referenceBuilder.h(qRef[0]);
  qRef[0] = referenceBuilder.z(qRef[0]);
  qRef[0] = referenceBuilder.h(qRef[0]);
  referenceBuilder.gphase(std::numbers::pi);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks if a multi-qubit phase gate is removed if it
 * only adds a global phase that is one.
 */
TEST_F(QCOConstantPropagationTest, testRemoveMultiQubitPhaseGatePlusOne) {
  auto q = programBuilder.allocQubitRegister(2);
  q[0] = programBuilder.h(q[0]);
  programBuilder.cz(q[0], q[1]);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  qRef[0] = referenceBuilder.h(qRef[0]);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks if a multi-qubit phase gate is replaced if it
 * only adds a global phase.
 */
TEST_F(QCOConstantPropagationTest, testRemoveMultiQubitPhaseGateMinusOne) {
  auto q = programBuilder.allocQubitRegister(2);
  q[0] = programBuilder.x(q[0]);
  auto [q0, q1] = programBuilder.cx(q[0], q[1]);
  programBuilder.cz(q0, q1);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  referenceBuilder.x(qRef[0]);
  referenceBuilder.x(qRef[1]);
  referenceBuilder.gphase(std::numbers::pi);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks if a multi-qubit phase gate is not removed if
 * it adds different global phases depending on the actual state.
 */
TEST_F(QCOConstantPropagationTest, testDoNotRemoveMultiQubitPhaseGate) {
  auto q = programBuilder.allocQubitRegister(2);
  q[0] = programBuilder.h(q[0]);
  auto [q0, q1] = programBuilder.cx(q[0], q[1]);
  programBuilder.cz(q0, q1);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(2);
  qRef[0] = referenceBuilder.h(qRef[0]);
  auto [qRef0, qRef1] = referenceBuilder.cx(qRef[0], qRef[1]);
  referenceBuilder.cz(qRef0, qRef1);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}

/**
 * @brief Test: This test checks that a quantum conditional is replaced by a
 * classical if a qubit and a classical bit are equivalent.
 */
TEST_F(QCOConstantPropagationTest, testMoveMeasurementToFront) {
  auto q = programBuilder.allocQubitRegister(3);
  q[0] = programBuilder.h(q[0]);
  auto [q0, q1] = programBuilder.cx(q[0], q[1]);
  auto [q11, q2] = programBuilder.cx(q1, q[2]);
  programBuilder.h(q11);
  programBuilder.measure(q0);
  module = programBuilder.finalize();

  auto qRef = referenceBuilder.allocQubitRegister(3);
  qRef[0] = referenceBuilder.h(qRef[0]);
  auto [qRef0, qRef1] = referenceBuilder.cx(qRef[0], qRef[1]);
  auto [qRef01, bRef0] = referenceBuilder.measure(qRef0);
  referenceBuilder.qcoIf(bRef0, {qRef[2]}, [&](const ValueRange args) {
    const auto qi0 = referenceBuilder.x(args[0]);
    return SmallVector{qi0};
  });
  referenceBuilder.h(qRef1);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runConstantPropagationPass(module.get()).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reference.get()));
}
