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
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
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

class QCOQuantumIPOTest : public testing::Test {

protected:
  MLIRContext context;
  QCOProgramBuilder programBuilder;
  QCOProgramBuilder referenceBuilder;
  OwningOpRef<ModuleOp> module;
  OwningOpRef<ModuleOp> reference;

  QCOQuantumIPOTest() : programBuilder(&context), referenceBuilder(&context) {}

  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect,
                    qtensor::QTensorDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  /**
   * @brief Adds the quantum IPO pass to the current context and runs it.
   *
   * @param module The module to run the pass on.
   */
  static LogicalResult runQuantumIPOPass(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(createQuantumIPO());
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

  /**
   * @brief Runs the pass on the constructed module and compares it against the
   * constructed reference.
   */
  void expectModuleMatchesReference() {
    ASSERT_TRUE(runQuantumIPOPass(module.get()).succeeded());
    ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

    EXPECT_TRUE(
        areModulesEquivalentWithPermutations(module.get(), reference.get()));
  }
};

} // namespace

// ==========================================================================
// Context-sensitive specialization for arguments in the |0> state.
// ==========================================================================

/**
 * @brief A gate that acts trivially on |0> is dropped from a specialized copy
 * of the callee when the caller passes a freshly allocated qubit.
 */
TEST_F(QCOQuantumIPOTest, specializeZeroArgumentDropsDiagonalGate) {
  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {programBuilder.getQubitType()},
                                           {programBuilder.getQubitType()});
  programBuilder.endFunction({programBuilder.z(args[0])});

  auto q = programBuilder.allocQubit();
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  // The original callee is retained, ...
  auto refArgs =
      referenceBuilder.startFunction("f", {referenceBuilder.getQubitType()},
                                     {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction({referenceBuilder.z(refArgs[0])});
  // ... while the call is redirected to a specialization without the gate.
  auto specArgs = referenceBuilder.startFunction(
      "f_spec_zero_arg_0", {referenceBuilder.getQubitType()},
      {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction({specArgs[0]});

  auto refQ = referenceBuilder.allocQubit();
  auto refResults = referenceBuilder.call("f_spec_zero_arg_0", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief A controlled gate whose control is known to be in the |0> state is
 * dropped entirely, together with its effect on the target qubit.
 */
TEST_F(QCOQuantumIPOTest, specializeZeroArgumentDropsControlledGate) {
  const auto qubitType = programBuilder.getQubitType();

  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {qubitType, qubitType},
                                           {qubitType, qubitType});
  auto control = args[0];
  auto target = args[1];
  std::tie(control, target) = programBuilder.cx(control, target);
  programBuilder.endFunction({control, target});

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.h(programBuilder.allocQubit());
  auto results = programBuilder.call("f", {q0, q1});
  programBuilder.sink(results[0]);
  programBuilder.sink(results[1]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs = referenceBuilder.startFunction("f", {qubitType, qubitType},
                                                {qubitType, qubitType});
  auto refControl = refArgs[0];
  auto refTarget = refArgs[1];
  std::tie(refControl, refTarget) = referenceBuilder.cx(refControl, refTarget);
  referenceBuilder.endFunction({refControl, refTarget});

  auto specArgs = referenceBuilder.startFunction(
      "f_spec_zero_arg_0", {qubitType, qubitType}, {qubitType, qubitType});
  referenceBuilder.endFunction({specArgs[0], specArgs[1]});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.h(referenceBuilder.allocQubit());
  auto refResults = referenceBuilder.call("f_spec_zero_arg_0", {refQ0, refQ1});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief A gate that does not act trivially on |0> must not be dropped.
 */
TEST_F(QCOQuantumIPOTest, noZeroSpecializationForNonTrivialGate) {
  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {programBuilder.getQubitType()},
                                           {programBuilder.getQubitType()});
  programBuilder.endFunction({programBuilder.x(args[0])});

  auto q = programBuilder.allocQubit();
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {referenceBuilder.getQubitType()},
                                     {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction({referenceBuilder.x(refArgs[0])});

  auto refQ = referenceBuilder.allocQubit();
  auto refResults = referenceBuilder.call("f", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief If the state of the argument is unknown, no specialization applies.
 */
TEST_F(QCOQuantumIPOTest, noSpecializationForUnknownArgumentState) {
  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {programBuilder.getQubitType()},
                                           {programBuilder.getQubitType()});
  programBuilder.endFunction({programBuilder.z(args[0])});

  // A `y` gate leaves the qubit in a state the pass cannot reason about.
  auto q = programBuilder.y(programBuilder.allocQubit());
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {referenceBuilder.getQubitType()},
                                     {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction({referenceBuilder.z(refArgs[0])});

  auto refQ = referenceBuilder.y(referenceBuilder.allocQubit());
  auto refResults = referenceBuilder.call("f", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief Two call sites that qualify for the same specialization share a single
 * specialized copy of the callee.
 */
TEST_F(QCOQuantumIPOTest, reuseZeroSpecializationAcrossCallSites) {
  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {programBuilder.getQubitType()},
                                           {programBuilder.getQubitType()});
  programBuilder.endFunction({programBuilder.s(args[0])});

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto results0 = programBuilder.call("f", {q0});
  auto results1 = programBuilder.call("f", {q1});
  programBuilder.sink(results0[0]);
  programBuilder.sink(results1[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {referenceBuilder.getQubitType()},
                                     {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction({referenceBuilder.s(refArgs[0])});

  auto specArgs = referenceBuilder.startFunction(
      "f_spec_zero_arg_0", {referenceBuilder.getQubitType()},
      {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction({specArgs[0]});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refResults0 = referenceBuilder.call("f_spec_zero_arg_0", {refQ0});
  auto refResults1 = referenceBuilder.call("f_spec_zero_arg_0", {refQ1});
  referenceBuilder.sink(refResults0[0]);
  referenceBuilder.sink(refResults1[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

// ==========================================================================
// Context-sensitive specialization for arguments in the |+> state.
// ==========================================================================

/**
 * @brief An `x` gate acting on a qubit known to be in the |+> state is dropped
 * from a specialized copy of the callee.
 */
TEST_F(QCOQuantumIPOTest, specializePlusArgumentDropsXGate) {
  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {programBuilder.getQubitType()},
                                           {programBuilder.getQubitType()});
  programBuilder.endFunction({programBuilder.x(args[0])});

  auto q = programBuilder.h(programBuilder.allocQubit());
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {referenceBuilder.getQubitType()},
                                     {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction({referenceBuilder.x(refArgs[0])});

  auto specArgs = referenceBuilder.startFunction(
      "f_spec_plus_arg_0", {referenceBuilder.getQubitType()},
      {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction({specArgs[0]});

  auto refQ = referenceBuilder.h(referenceBuilder.allocQubit());
  auto refResults = referenceBuilder.call("f_spec_plus_arg_0", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief A gate that does not act trivially on |+> must not be dropped.
 */
TEST_F(QCOQuantumIPOTest, noPlusSpecializationForNonXGate) {
  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {programBuilder.getQubitType()},
                                           {programBuilder.getQubitType()});
  programBuilder.endFunction({programBuilder.z(args[0])});

  auto q = programBuilder.h(programBuilder.allocQubit());
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {referenceBuilder.getQubitType()},
                                     {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction({referenceBuilder.z(refArgs[0])});

  auto refQ = referenceBuilder.h(referenceBuilder.allocQubit());
  auto refResults = referenceBuilder.call("f", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

// ==========================================================================
// Context-sensitive specialization for constant rotation angles.
// ==========================================================================

/**
 * @brief A rotation angle of pi passed at the call site is baked into a
 * specialized copy of the callee.
 */
TEST_F(QCOQuantumIPOTest, specializeConstantRotationAngle) {
  const auto qubitType = programBuilder.getQubitType();
  const auto floatType = programBuilder.getF64Type();

  programBuilder.initialize();
  auto args =
      programBuilder.startFunction("f", {qubitType, floatType}, {qubitType});
  programBuilder.endFunction({programBuilder.rz(args[1], args[0])});

  auto q = programBuilder.allocQubit();
  auto angle = programBuilder.floatConstant(std::numbers::pi);
  auto results = programBuilder.call("f", {q, angle});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {qubitType, floatType}, {qubitType});
  referenceBuilder.endFunction({referenceBuilder.rz(refArgs[1], refArgs[0])});

  // The specialized copy keeps the parameter in its signature but no longer
  // reads it; the angle becomes a constant in the body.
  auto specArgs = referenceBuilder.startFunction(
      "f_spec_fixed_angle_1", {qubitType, floatType}, {qubitType});
  referenceBuilder.endFunction(
      {referenceBuilder.rz(std::numbers::pi, specArgs[0])});

  auto refQ = referenceBuilder.allocQubit();
  auto refAngle = referenceBuilder.floatConstant(std::numbers::pi);
  auto refResults =
      referenceBuilder.call("f_spec_fixed_angle_1", {refQ, refAngle});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief A rotation angle of pi/2 is likewise specialized.
 */
TEST_F(QCOQuantumIPOTest, specializeHalfPiRotationAngle) {
  const auto qubitType = programBuilder.getQubitType();
  const auto floatType = programBuilder.getF64Type();

  programBuilder.initialize();
  auto args =
      programBuilder.startFunction("f", {qubitType, floatType}, {qubitType});
  programBuilder.endFunction({programBuilder.rx(args[1], args[0])});

  auto q = programBuilder.allocQubit();
  auto angle = programBuilder.floatConstant(std::numbers::pi / 2);
  auto results = programBuilder.call("f", {q, angle});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {qubitType, floatType}, {qubitType});
  referenceBuilder.endFunction({referenceBuilder.rx(refArgs[1], refArgs[0])});

  auto specArgs = referenceBuilder.startFunction(
      "f_spec_fixed_angle_1", {qubitType, floatType}, {qubitType});
  referenceBuilder.endFunction(
      {referenceBuilder.rx(std::numbers::pi / 2, specArgs[0])});

  auto refQ = referenceBuilder.allocQubit();
  auto refAngle = referenceBuilder.floatConstant(std::numbers::pi / 2);
  auto refResults =
      referenceBuilder.call("f_spec_fixed_angle_1", {refQ, refAngle});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief An angle outside the set of specialized angles leaves the callee
 * untouched.
 */
TEST_F(QCOQuantumIPOTest, noSpecializationForArbitraryRotationAngle) {
  const auto qubitType = programBuilder.getQubitType();
  const auto floatType = programBuilder.getF64Type();

  programBuilder.initialize();
  auto args =
      programBuilder.startFunction("f", {qubitType, floatType}, {qubitType});
  programBuilder.endFunction({programBuilder.rz(args[1], args[0])});

  auto q = programBuilder.allocQubit();
  auto angle = programBuilder.floatConstant(0.7);
  auto results = programBuilder.call("f", {q, angle});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {qubitType, floatType}, {qubitType});
  referenceBuilder.endFunction({referenceBuilder.rz(refArgs[1], refArgs[0])});

  auto refQ = referenceBuilder.allocQubit();
  auto refAngle = referenceBuilder.floatConstant(0.7);
  auto refResults = referenceBuilder.call("f", {refQ, refAngle});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief Two call sites that qualify for the same |+> specialization share a
 * single specialized copy of the callee.
 */
TEST_F(QCOQuantumIPOTest, reusePlusSpecializationAcrossCallSites) {
  const auto qubitType = programBuilder.getQubitType();

  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {qubitType}, {qubitType});
  programBuilder.endFunction({programBuilder.x(args[0])});

  auto q0 = programBuilder.h(programBuilder.allocQubit());
  auto q1 = programBuilder.h(programBuilder.allocQubit());
  auto results0 = programBuilder.call("f", {q0});
  auto results1 = programBuilder.call("f", {q1});
  programBuilder.sink(results0[0]);
  programBuilder.sink(results1[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs = referenceBuilder.startFunction("f", {qubitType}, {qubitType});
  referenceBuilder.endFunction({referenceBuilder.x(refArgs[0])});
  auto specArgs = referenceBuilder.startFunction("f_spec_plus_arg_0",
                                                 {qubitType}, {qubitType});
  referenceBuilder.endFunction({specArgs[0]});

  auto refQ0 = referenceBuilder.h(referenceBuilder.allocQubit());
  auto refQ1 = referenceBuilder.h(referenceBuilder.allocQubit());
  auto refResults0 = referenceBuilder.call("f_spec_plus_arg_0", {refQ0});
  auto refResults1 = referenceBuilder.call("f_spec_plus_arg_0", {refQ1});
  referenceBuilder.sink(refResults0[0]);
  referenceBuilder.sink(refResults1[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief Two call sites passing the same constant angle share a single
 * specialized copy of the callee.
 */
TEST_F(QCOQuantumIPOTest, reuseRotationSpecializationAcrossCallSites) {
  const auto qubitType = programBuilder.getQubitType();
  const auto floatType = programBuilder.getF64Type();

  programBuilder.initialize();
  auto args =
      programBuilder.startFunction("f", {qubitType, floatType}, {qubitType});
  programBuilder.endFunction({programBuilder.rz(args[1], args[0])});

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto angle = programBuilder.floatConstant(std::numbers::pi);
  auto results0 = programBuilder.call("f", {q0, angle});
  auto results1 = programBuilder.call("f", {q1, angle});
  programBuilder.sink(results0[0]);
  programBuilder.sink(results1[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {qubitType, floatType}, {qubitType});
  referenceBuilder.endFunction({referenceBuilder.rz(refArgs[1], refArgs[0])});
  auto specArgs = referenceBuilder.startFunction(
      "f_spec_fixed_angle_1", {qubitType, floatType}, {qubitType});
  referenceBuilder.endFunction(
      {referenceBuilder.rz(std::numbers::pi, specArgs[0])});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refAngle = referenceBuilder.floatConstant(std::numbers::pi);
  auto refResults0 =
      referenceBuilder.call("f_spec_fixed_angle_1", {refQ0, refAngle});
  auto refResults1 =
      referenceBuilder.call("f_spec_fixed_angle_1", {refQ1, refAngle});
  referenceBuilder.sink(refResults0[0]);
  referenceBuilder.sink(refResults1[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

// ==========================================================================
// Quantum argument promotion.
// ==========================================================================

/**
 * @brief A tensor argument whose elements are extracted and re-inserted at
 * compile-time constant indices is replaced by scalar qubit arguments.
 */
TEST_F(QCOQuantumIPOTest, promoteTensorArgumentToQubitArgument) {
  const auto tensorType = programBuilder.getQubitTensorType(2);

  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {tensorType}, {tensorType});
  auto [tensorIn, inner] = programBuilder.qtensorExtract(args[0], 0);
  inner = programBuilder.h(inner);
  programBuilder.endFunction(
      {programBuilder.qtensorInsert(inner, tensorIn, 0)});

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto tensor = programBuilder.qtensorFromElements({q0, q1});
  auto results = programBuilder.call("f", {tensor});
  programBuilder.qtensorDealloc(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {referenceBuilder.getQubitType()},
                                     {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction({referenceBuilder.h(refArgs[0])});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refTensor = referenceBuilder.qtensorFromElements({refQ0, refQ1});
  // The caller extracts the promoted element, calls, and re-inserts it.
  auto [refTensorIn, refExtracted] =
      referenceBuilder.qtensorExtract(refTensor, 0);
  auto refResults = referenceBuilder.call("f", {refExtracted});
  auto refInserted =
      referenceBuilder.qtensorInsert(refResults[0], refTensorIn, 0);
  referenceBuilder.qtensorDealloc(refInserted);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief Only the tensor elements the callee actually touches become scalar
 * arguments; untouched elements never cross the call boundary.
 */
TEST_F(QCOQuantumIPOTest, promoteOnlyUsedTensorElements) {
  const auto tensorType = programBuilder.getQubitTensorType(3);

  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {tensorType}, {tensorType});
  auto [tensorIn, inner] = programBuilder.qtensorExtract(args[0], 1);
  inner = programBuilder.x(inner);
  programBuilder.endFunction(
      {programBuilder.qtensorInsert(inner, tensorIn, 1)});

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.allocQubit();
  auto tensor = programBuilder.qtensorFromElements({q0, q1, q2});
  auto results = programBuilder.call("f", {tensor});
  programBuilder.qtensorDealloc(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {referenceBuilder.getQubitType()},
                                     {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction({referenceBuilder.x(refArgs[0])});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refQ2 = referenceBuilder.allocQubit();
  auto refTensor = referenceBuilder.qtensorFromElements({refQ0, refQ1, refQ2});
  auto [refTensorIn, refExtracted] =
      referenceBuilder.qtensorExtract(refTensor, 1);
  auto refResults = referenceBuilder.call("f", {refExtracted});
  auto refInserted =
      referenceBuilder.qtensorInsert(refResults[0], refTensorIn, 1);
  referenceBuilder.qtensorDealloc(refInserted);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief A qubit that is moved to a different slot is promoted with the
 * extraction and insertion indices kept apart.
 */
TEST_F(QCOQuantumIPOTest, promoteTensorElementIntoDifferentSlot) {
  const auto tensorType = programBuilder.getQubitTensorType(2);

  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {tensorType}, {tensorType});
  auto [tensorIn, inner] = programBuilder.qtensorExtract(args[0], 0);
  inner = programBuilder.h(inner);
  programBuilder.endFunction(
      {programBuilder.qtensorInsert(inner, tensorIn, 1)});

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto tensor = programBuilder.qtensorFromElements({q0, q1});
  auto results = programBuilder.call("f", {tensor});
  programBuilder.qtensorDealloc(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {referenceBuilder.getQubitType()},
                                     {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction({referenceBuilder.h(refArgs[0])});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refTensor = referenceBuilder.qtensorFromElements({refQ0, refQ1});
  auto [refTensorIn, refExtracted] =
      referenceBuilder.qtensorExtract(refTensor, 0);
  auto refResults = referenceBuilder.call("f", {refExtracted});
  auto refInserted =
      referenceBuilder.qtensorInsert(refResults[0], refTensorIn, 1);
  referenceBuilder.qtensorDealloc(refInserted);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief An element that is extracted but never re-inserted cannot be promoted,
 * because the promoted callee would have nothing to hand back for that slot.
 */
TEST_F(QCOQuantumIPOTest, noPromotionWithoutMatchingInsert) {
  const auto tensorType = programBuilder.getQubitTensorType(2);
  const auto qubitType = programBuilder.getQubitType();

  programBuilder.initialize();
  auto args =
      programBuilder.startFunction("f", {tensorType}, {tensorType, qubitType});
  // The element at index 0 leaves the tensor for good.
  auto [tensorIn, escaping] = programBuilder.qtensorExtract(args[0], 0);
  escaping = programBuilder.h(escaping);
  programBuilder.endFunction({tensorIn, escaping});

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto tensor = programBuilder.qtensorFromElements({q0, q1});
  auto results = programBuilder.call("f", {tensor});
  programBuilder.sink(results[1]);
  programBuilder.qtensorDealloc(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs = referenceBuilder.startFunction("f", {tensorType},
                                                {tensorType, qubitType});
  auto [refTensorIn, refEscaping] =
      referenceBuilder.qtensorExtract(refArgs[0], 0);
  refEscaping = referenceBuilder.h(refEscaping);
  referenceBuilder.endFunction({refTensorIn, refEscaping});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refTensor = referenceBuilder.qtensorFromElements({refQ0, refQ1});
  auto refResults = referenceBuilder.call("f", {refTensor});
  referenceBuilder.sink(refResults[1]);
  referenceBuilder.qtensorDealloc(refResults[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief A tensor argument that never has an element taken out of it has
 * nothing to promote.
 */
TEST_F(QCOQuantumIPOTest, noPromotionWithoutElementAccess) {
  const auto tensorType = programBuilder.getQubitTensorType(2);

  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {tensorType}, {tensorType});
  programBuilder.endFunction({args[0]});

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto tensor = programBuilder.qtensorFromElements({q0, q1});
  auto results = programBuilder.call("f", {tensor});
  programBuilder.qtensorDealloc(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {tensorType}, {tensorType});
  referenceBuilder.endFunction({refArgs[0]});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refTensor = referenceBuilder.qtensorFromElements({refQ0, refQ1});
  auto refResults = referenceBuilder.call("f", {refTensor});
  referenceBuilder.qtensorDealloc(refResults[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief A callee that touches several tensor elements gets one scalar argument
 * and one scalar result per element.
 */
TEST_F(QCOQuantumIPOTest, promoteMultipleTensorElements) {
  const auto tensorType = programBuilder.getQubitTensorType(2);

  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {tensorType}, {tensorType});
  auto [afterFirst, first] = programBuilder.qtensorExtract(args[0], 0);
  auto firstTensor =
      programBuilder.qtensorInsert(programBuilder.h(first), afterFirst, 0);
  auto [afterSecond, second] = programBuilder.qtensorExtract(firstTensor, 1);
  programBuilder.endFunction(
      {programBuilder.qtensorInsert(programBuilder.x(second), afterSecond, 1)});

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto tensor = programBuilder.qtensorFromElements({q0, q1});
  auto results = programBuilder.call("f", {tensor});
  programBuilder.qtensorDealloc(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  const auto qubitType = referenceBuilder.getQubitType();
  auto refArgs = referenceBuilder.startFunction("f", {qubitType, qubitType},
                                                {qubitType, qubitType});
  referenceBuilder.endFunction(
      {referenceBuilder.h(refArgs[0]), referenceBuilder.x(refArgs[1])});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refTensor = referenceBuilder.qtensorFromElements({refQ0, refQ1});
  // The caller takes every promoted element out before the call and puts them
  // all back afterwards.
  auto [refAfterFirst, refFirst] =
      referenceBuilder.qtensorExtract(refTensor, 0);
  auto [refAfterSecond, refSecond] =
      referenceBuilder.qtensorExtract(refAfterFirst, 1);
  auto refResults = referenceBuilder.call("f", {refFirst, refSecond});
  auto refFirstBack =
      referenceBuilder.qtensorInsert(refResults[0], refAfterSecond, 0);
  referenceBuilder.qtensorDealloc(
      referenceBuilder.qtensorInsert(refResults[1], refFirstBack, 1));
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief A promoted element may be measured inside the callee; the measurement
 * outcome stays a separate result and the caller keeps reading it.
 */
TEST_F(QCOQuantumIPOTest, promoteTensorElementWithMeasurement) {
  const auto tensorType = programBuilder.getQubitTensorType(2);
  const auto bitType = programBuilder.getI1Type();

  programBuilder.initialize({bitType});
  auto args =
      programBuilder.startFunction("f", {tensorType}, {tensorType, bitType});
  auto [rest, inner] = programBuilder.qtensorExtract(args[0], 0);
  Value bit;
  std::tie(inner, bit) = programBuilder.measure(inner);
  programBuilder.endFunction(
      {programBuilder.qtensorInsert(inner, rest, 0), bit});

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto tensor = programBuilder.qtensorFromElements({q0, q1});
  auto results = programBuilder.call("f", {tensor});
  programBuilder.qtensorDealloc(results[0]);
  module = programBuilder.finalize({results[1]});

  referenceBuilder.initialize({bitType});
  auto refArgs = referenceBuilder.startFunction(
      "f", {referenceBuilder.getQubitType()},
      {referenceBuilder.getQubitType(), bitType});
  Value refBit;
  auto refInner = refArgs[0];
  std::tie(refInner, refBit) = referenceBuilder.measure(refInner);
  referenceBuilder.endFunction({refInner, refBit});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refTensor = referenceBuilder.qtensorFromElements({refQ0, refQ1});
  auto [refRest, refExtracted] = referenceBuilder.qtensorExtract(refTensor, 0);
  auto refResults = referenceBuilder.call("f", {refExtracted});
  referenceBuilder.qtensorDealloc(
      referenceBuilder.qtensorInsert(refResults[0], refRest, 0));
  reference = referenceBuilder.finalize({refResults[1]});

  expectModuleMatchesReference();
}

/**
 * @brief The tensor has to be handed back as the first result, because that is
 * the result the promoted qubits take the place of.
 */
TEST_F(QCOQuantumIPOTest, noPromotionWhenTensorIsNotFirstResult) {
  const auto tensorType = programBuilder.getQubitTensorType(2);
  const auto bitType = programBuilder.getI1Type();

  const auto buildProgram = [&](QCOProgramBuilder& b) {
    b.initialize({bitType});
    auto args = b.startFunction("f", {tensorType}, {bitType, tensorType});
    auto [rest, inner] = b.qtensorExtract(args[0], 0);
    Value bit;
    std::tie(inner, bit) = b.measure(inner);
    b.endFunction({bit, b.qtensorInsert(inner, rest, 0)});

    auto q0 = b.allocQubit();
    auto q1 = b.allocQubit();
    auto tensor = b.qtensorFromElements({q0, q1});
    auto results = b.call("f", {tensor});
    b.qtensorDealloc(results[1]);
    return results[0];
  };

  module = programBuilder.finalize({buildProgram(programBuilder)});
  reference = referenceBuilder.finalize({buildProgram(referenceBuilder)});

  expectModuleMatchesReference();
}

// ==========================================================================
// Auxiliary qubit hoisting.
// ==========================================================================

/**
 * @brief A qubit that a callee allocates and releases internally is turned into
 * an extra argument, so the caller owns the allocation and can reuse it.
 */
TEST_F(QCOQuantumIPOTest, hoistAuxiliaryQubitIntoCaller) {
  const auto qubitType = programBuilder.getQubitType();

  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {qubitType}, {qubitType});
  auto aux = programBuilder.allocQubit();
  auto target = args[0];
  std::tie(aux, target) = programBuilder.cx(aux, target);
  programBuilder.sink(aux);
  programBuilder.endFunction({target});

  auto q = programBuilder.allocQubit();
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  // The auxiliary qubit becomes a trailing argument and is returned in a reset
  // state as a trailing result.
  auto refArgs = referenceBuilder.startFunction("f", {qubitType, qubitType},
                                                {qubitType, qubitType});
  auto refAux = refArgs[1];
  auto refTarget = refArgs[0];
  std::tie(refAux, refTarget) = referenceBuilder.cx(refAux, refTarget);
  refAux = referenceBuilder.reset(refAux);
  referenceBuilder.endFunction({refTarget, refAux});

  auto refQ = referenceBuilder.allocQubit();
  auto refAuxAlloc = referenceBuilder.allocQubit();
  auto refResults = referenceBuilder.call("f", {refQ, refAuxAlloc});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief A qubit that the callee allocates but hands back to the caller is not
 * auxiliary and must stay where it is.
 */
TEST_F(QCOQuantumIPOTest, noHoistingForReturnedQubit) {
  const auto qubitType = programBuilder.getQubitType();

  programBuilder.initialize();
  auto args =
      programBuilder.startFunction("f", {qubitType}, {qubitType, qubitType});
  auto fresh = programBuilder.allocQubit();
  programBuilder.endFunction({args[0], fresh});

  auto q = programBuilder.allocQubit();
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  programBuilder.sink(results[1]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {qubitType}, {qubitType, qubitType});
  auto refFresh = referenceBuilder.allocQubit();
  referenceBuilder.endFunction({refArgs[0], refFresh});

  auto refQ = referenceBuilder.allocQubit();
  auto refResults = referenceBuilder.call("f", {refQ});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief The auxiliary qubit is tracked across a measurement and a reset on its
 * way to the release point.
 *
 * The measurement outcome is handed back to the caller so that the measurement
 * is not dead, and the reset sits between two gates so that it is neither
 * folded into the allocation nor into the release.
 */
TEST_F(QCOQuantumIPOTest, hoistAuxiliaryQubitThroughMeasureAndReset) {
  const auto qubitType = programBuilder.getQubitType();
  const auto bitType = programBuilder.getI1Type();

  programBuilder.initialize({bitType});
  auto args =
      programBuilder.startFunction("f", {qubitType}, {qubitType, bitType});
  auto aux = programBuilder.h(programBuilder.allocQubit());
  Value bit;
  std::tie(aux, bit) = programBuilder.measure(aux);
  aux = programBuilder.reset(aux);
  auto target = args[0];
  std::tie(aux, target) = programBuilder.cx(aux, target);
  programBuilder.sink(aux);
  programBuilder.endFunction({target, bit});

  auto q = programBuilder.allocQubit();
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize({results[1]});

  referenceBuilder.initialize({bitType});
  auto refArgs = referenceBuilder.startFunction(
      "f", {qubitType, qubitType}, {qubitType, bitType, qubitType});
  auto refAux = referenceBuilder.h(refArgs[1]);
  Value refBit;
  std::tie(refAux, refBit) = referenceBuilder.measure(refAux);
  refAux = referenceBuilder.reset(refAux);
  auto refTarget = refArgs[0];
  std::tie(refAux, refTarget) = referenceBuilder.cx(refAux, refTarget);
  refAux = referenceBuilder.reset(refAux);
  referenceBuilder.endFunction({refTarget, refBit, refAux});

  auto refQ = referenceBuilder.allocQubit();
  auto refAuxAlloc = referenceBuilder.allocQubit();
  auto refResults = referenceBuilder.call("f", {refQ, refAuxAlloc});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[2]);
  reference = referenceBuilder.finalize({refResults[1]});

  expectModuleMatchesReference();
}

/**
 * @brief The auxiliary qubit is tracked while it is parked in a tensor, past
 * an extraction of an unrelated element.
 */
TEST_F(QCOQuantumIPOTest, hoistAuxiliaryQubitThroughTensor) {
  const auto qubitType = programBuilder.getQubitType();

  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {qubitType}, {qubitType});
  auto aux = programBuilder.allocQubit();
  auto target = args[0];
  std::tie(aux, target) = programBuilder.cx(aux, target);
  // The auxiliary qubit sits at index 0, the argument qubit at index 1.
  auto tensor = programBuilder.qtensorFromElements({aux, target});
  auto [afterOther, other] = programBuilder.qtensorExtract(tensor, 1);
  auto [afterAux, auxBack] = programBuilder.qtensorExtract(afterOther, 0);
  programBuilder.sink(auxBack);
  programBuilder.qtensorDealloc(afterAux);
  programBuilder.endFunction({other});

  auto q = programBuilder.allocQubit();
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs = referenceBuilder.startFunction("f", {qubitType, qubitType},
                                                {qubitType, qubitType});
  auto refAux = refArgs[1];
  auto refTarget = refArgs[0];
  std::tie(refAux, refTarget) = referenceBuilder.cx(refAux, refTarget);
  auto refTensor = referenceBuilder.qtensorFromElements({refAux, refTarget});
  auto [refAfterOther, refOther] =
      referenceBuilder.qtensorExtract(refTensor, 1);
  auto [refAfterAux, refAuxBack] =
      referenceBuilder.qtensorExtract(refAfterOther, 0);
  auto refReset = referenceBuilder.reset(refAuxBack);
  referenceBuilder.qtensorDealloc(refAfterAux);
  referenceBuilder.endFunction({refOther, refReset});

  auto refQ = referenceBuilder.allocQubit();
  auto refAuxAlloc = referenceBuilder.allocQubit();
  auto refResults = referenceBuilder.call("f", {refQ, refAuxAlloc});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief The auxiliary qubit is tracked across a nested call on its way to the
 * release point.
 *
 * The nested callee returns more than one qubit and the auxiliary one is not
 * the first, so the walk has to match the operand position rather than simply
 * taking the first result.
 */
TEST_F(QCOQuantumIPOTest, hoistAuxiliaryQubitThroughNestedCall) {
  const auto qubitType = programBuilder.getQubitType();

  const auto buildNestedCallee = [&qubitType](QCOProgramBuilder& b) {
    auto innerArgs =
        b.startFunction("g", {qubitType, qubitType}, {qubitType, qubitType});
    b.endFunction({b.h(innerArgs[0]), innerArgs[1]});
  };

  programBuilder.initialize();
  buildNestedCallee(programBuilder);

  auto args = programBuilder.startFunction("f", {qubitType}, {qubitType});
  auto aux = programBuilder.allocQubit();
  // The auxiliary qubit is the second operand and the second result.
  auto nested = programBuilder.call("g", {args[0], aux});
  auto target = nested[0];
  aux = nested[1];
  std::tie(aux, target) = programBuilder.cx(aux, target);
  programBuilder.sink(aux);
  programBuilder.endFunction({target});

  auto q = programBuilder.allocQubit();
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  buildNestedCallee(referenceBuilder);

  auto refArgs = referenceBuilder.startFunction("f", {qubitType, qubitType},
                                                {qubitType, qubitType});
  auto refNested = referenceBuilder.call("g", {refArgs[0], refArgs[1]});
  auto refTarget = refNested[0];
  auto refAux = refNested[1];
  std::tie(refAux, refTarget) = referenceBuilder.cx(refAux, refTarget);
  refAux = referenceBuilder.reset(refAux);
  referenceBuilder.endFunction({refTarget, refAux});

  auto refQ = referenceBuilder.allocQubit();
  auto refAuxAlloc = referenceBuilder.allocQubit();
  auto refResults = referenceBuilder.call("f", {refQ, refAuxAlloc});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief Every call site of a hoisted callee gets its own allocation.
 */
TEST_F(QCOQuantumIPOTest, hoistAuxiliaryQubitWithMultipleCallSites) {
  const auto qubitType = programBuilder.getQubitType();

  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {qubitType}, {qubitType});
  auto aux = programBuilder.allocQubit();
  auto target = args[0];
  std::tie(aux, target) = programBuilder.cx(aux, target);
  programBuilder.sink(aux);
  programBuilder.endFunction({target});

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto results0 = programBuilder.call("f", {q0});
  auto results1 = programBuilder.call("f", {q1});
  programBuilder.sink(results0[0]);
  programBuilder.sink(results1[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs = referenceBuilder.startFunction("f", {qubitType, qubitType},
                                                {qubitType, qubitType});
  auto refAux = refArgs[1];
  auto refTarget = refArgs[0];
  std::tie(refAux, refTarget) = referenceBuilder.cx(refAux, refTarget);
  refAux = referenceBuilder.reset(refAux);
  referenceBuilder.endFunction({refTarget, refAux});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refAux0 = referenceBuilder.allocQubit();
  auto refResults0 = referenceBuilder.call("f", {refQ0, refAux0});
  referenceBuilder.sink(refResults0[1]);
  auto refAux1 = referenceBuilder.allocQubit();
  auto refResults1 = referenceBuilder.call("f", {refQ1, refAux1});
  referenceBuilder.sink(refResults1[1]);
  referenceBuilder.sink(refResults0[0]);
  referenceBuilder.sink(refResults1[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief A recursive callee is left alone, because its allocation would have to
 * be threaded through every level of the recursion. A caller of a recursive
 * function is still hoisted.
 */
TEST_F(QCOQuantumIPOTest, noHoistingForRecursiveFunction) {
  const auto qubitType = programBuilder.getQubitType();

  const auto buildRecursiveCallee = [&qubitType](QCOProgramBuilder& b) {
    auto innerArgs = b.startFunction("inner", {qubitType}, {qubitType});
    auto innerAux = b.allocQubit();
    auto innerTarget = innerArgs[0];
    std::tie(innerAux, innerTarget) = b.cx(innerAux, innerTarget);
    b.sink(innerAux);
    b.endFunction({b.call("inner", {innerTarget})[0]});
  };

  programBuilder.initialize();
  buildRecursiveCallee(programBuilder);

  auto args = programBuilder.startFunction("outer", {qubitType}, {qubitType});
  auto aux = programBuilder.allocQubit();
  auto target = args[0];
  std::tie(aux, target) = programBuilder.cx(aux, target);
  programBuilder.sink(aux);
  programBuilder.endFunction({programBuilder.call("inner", {target})[0]});

  auto q = programBuilder.allocQubit();
  auto results = programBuilder.call("outer", {q});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  // `inner` is recursive and therefore untouched, ...
  buildRecursiveCallee(referenceBuilder);

  // ... while `outer` is hoisted even though it calls a recursive function.
  auto refArgs = referenceBuilder.startFunction("outer", {qubitType, qubitType},
                                                {qubitType, qubitType});
  auto refAux = refArgs[1];
  auto refTarget = refArgs[0];
  std::tie(refAux, refTarget) = referenceBuilder.cx(refAux, refTarget);
  refAux = referenceBuilder.reset(refAux);
  referenceBuilder.endFunction(
      {referenceBuilder.call("inner", {refTarget})[0], refAux});

  auto refQ = referenceBuilder.allocQubit();
  auto refAuxAlloc = referenceBuilder.allocQubit();
  auto refResults = referenceBuilder.call("outer", {refQ, refAuxAlloc});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief An allocation nested inside a region is not hoisted, because it is not
 * executed on every path through the function.
 */
TEST_F(QCOQuantumIPOTest, noHoistingForAllocInsideRegion) {
  const auto qubitType = programBuilder.getQubitType();

  const auto buildProgram = [&qubitType](QCOProgramBuilder& b) {
    b.initialize();
    auto args = b.startFunction("f", {qubitType, b.getI1Type()}, {qubitType});
    auto result = b.qcoIf(args[1], args[0], [&](Value qubit) {
      auto aux = b.allocQubit();
      auto inner = qubit;
      std::tie(aux, inner) = b.cx(aux, inner);
      b.sink(aux);
      return inner;
    });
    b.endFunction({result});

    auto q = b.allocQubit();
    Value bit;
    std::tie(q, bit) = b.measure(q);
    auto results = b.call("f", {q, bit});
    b.sink(results[0]);
  };

  buildProgram(programBuilder);
  module = programBuilder.finalize();

  buildProgram(referenceBuilder);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief The auxiliary qubit is tracked when it enters a tensor through an
 * insertion and while unrelated elements are moved in and out around it.
 */
TEST_F(QCOQuantumIPOTest, hoistAuxiliaryQubitParkedInTensor) {
  const auto qubitType = programBuilder.getQubitType();

  const auto buildBody = [](QCOProgramBuilder& b, Value aux, Value target) {
    // Park the auxiliary qubit in a scratch register at index 0.
    auto scratch = b.qtensorAlloc(2);
    auto [afterPlaceholder, placeholder] = b.qtensorExtract(scratch, 0);
    b.sink(placeholder);
    auto parked = b.qtensorInsert(aux, afterPlaceholder, 0);
    // Move an unrelated element out and back in while the auxiliary qubit
    // stays parked at index 0.
    auto [afterOther, other] = b.qtensorExtract(parked, 1);
    auto restored = b.qtensorInsert(other, afterOther, 1);
    auto [afterAux, auxBack] = b.qtensorExtract(restored, 0);
    b.qtensorDealloc(afterAux);
    return std::pair<Value, Value>{auxBack, target};
  };

  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {qubitType}, {qubitType});
  auto aux = programBuilder.allocQubit();
  auto target = args[0];
  std::tie(aux, target) = programBuilder.cx(aux, target);
  auto [auxBack, finalTarget] = buildBody(programBuilder, aux, target);
  programBuilder.sink(auxBack);
  programBuilder.endFunction({finalTarget});

  auto q = programBuilder.allocQubit();
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs = referenceBuilder.startFunction("f", {qubitType, qubitType},
                                                {qubitType, qubitType});
  auto refAux = refArgs[1];
  auto refTarget = refArgs[0];
  std::tie(refAux, refTarget) = referenceBuilder.cx(refAux, refTarget);
  auto [refAuxBack, refFinalTarget] =
      buildBody(referenceBuilder, refAux, refTarget);
  referenceBuilder.endFunction(
      {refFinalTarget, referenceBuilder.reset(refAuxBack)});

  auto refQ = referenceBuilder.allocQubit();
  auto refAuxAlloc = referenceBuilder.allocQubit();
  auto refResults = referenceBuilder.call("f", {refQ, refAuxAlloc});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief A call that only consumes linear values, and one that only produces
 * them, keep the builder's tracking consistent.
 */
TEST_F(QCOQuantumIPOTest, callConsumesAndProducesLinearValues) {
  const auto qubitType = programBuilder.getQubitType();
  const auto tensorType = programBuilder.getQubitTensorType(2);

  const auto buildProgram = [&](QCOProgramBuilder& b) {
    b.initialize();
    // Allocate before declaring the helpers so that the function scope has to
    // remember the already-tracked values of the caller.
    auto q = b.allocQubit();
    auto scratch = b.qtensorAlloc(2);

    auto consumeArgs = b.startFunction("consume", {qubitType, tensorType}, {});
    b.sink(consumeArgs[0]);
    b.qtensorDealloc(consumeArgs[1]);
    b.endFunction({});

    b.startFunction("produce", {}, {tensorType});
    b.endFunction({b.qtensorAlloc(2)});

    b.call("consume", {q, scratch});
    auto produced = b.call("produce", {});
    b.qtensorDealloc(produced[0]);
  };

  buildProgram(programBuilder);
  module = programBuilder.finalize();
  buildProgram(referenceBuilder);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

// ==========================================================================
// Quantum function boundary commutation.
// ==========================================================================

/**
 * @brief A self-inverse gate applied right before a call cancels with the same
 * gate at the start of the callee.
 */
TEST_F(QCOQuantumIPOTest, cancelSelfInverseGateAcrossCallBoundary) {
  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {programBuilder.getQubitType()},
                                           {programBuilder.getQubitType()});
  programBuilder.endFunction({programBuilder.h(programBuilder.x(args[0]))});

  auto q = programBuilder.x(programBuilder.allocQubit());
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {referenceBuilder.getQubitType()},
                                     {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction(
      {referenceBuilder.h(referenceBuilder.x(refArgs[0]))});

  // Both the caller-side and the callee-side gate disappear.
  auto specArgs = referenceBuilder.startFunction(
      "f_spec_boundary_commutation", {referenceBuilder.getQubitType()},
      {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction({referenceBuilder.h(specArgs[0])});

  auto refQ = referenceBuilder.allocQubit();
  auto refResults =
      referenceBuilder.call("f_spec_boundary_commutation", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief Two different gates across the call boundary do not cancel.
 */
TEST_F(QCOQuantumIPOTest, noCancellationForDifferentGates) {
  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {programBuilder.getQubitType()},
                                           {programBuilder.getQubitType()});
  programBuilder.endFunction({programBuilder.y(args[0])});

  auto q = programBuilder.x(programBuilder.allocQubit());
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {referenceBuilder.getQubitType()},
                                     {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction({referenceBuilder.y(refArgs[0])});

  auto refQ = referenceBuilder.x(referenceBuilder.allocQubit());
  auto refResults = referenceBuilder.call("f", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief Controlled gates are out of scope for boundary commutation, even when
 * the same one appears on both sides of the call. Cancelling them would require
 * reasoning about the control qubits as well.
 */
TEST_F(QCOQuantumIPOTest, noCancellationForControlledGates) {
  const auto qubitType = programBuilder.getQubitType();

  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {qubitType, qubitType},
                                           {qubitType, qubitType});
  auto innerControl = args[0];
  auto innerTarget = args[1];
  std::tie(innerControl, innerTarget) =
      programBuilder.cx(innerControl, innerTarget);
  programBuilder.endFunction({innerControl, innerTarget});

  auto q0 = programBuilder.y(programBuilder.allocQubit());
  auto q1 = programBuilder.y(programBuilder.allocQubit());
  std::tie(q0, q1) = programBuilder.cx(q0, q1);
  auto results = programBuilder.call("f", {q0, q1});
  programBuilder.sink(results[0]);
  programBuilder.sink(results[1]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs = referenceBuilder.startFunction("f", {qubitType, qubitType},
                                                {qubitType, qubitType});
  auto refInnerControl = refArgs[0];
  auto refInnerTarget = refArgs[1];
  std::tie(refInnerControl, refInnerTarget) =
      referenceBuilder.cx(refInnerControl, refInnerTarget);
  referenceBuilder.endFunction({refInnerControl, refInnerTarget});

  auto refQ0 = referenceBuilder.y(referenceBuilder.allocQubit());
  auto refQ1 = referenceBuilder.y(referenceBuilder.allocQubit());
  std::tie(refQ0, refQ1) = referenceBuilder.cx(refQ0, refQ1);
  auto refResults = referenceBuilder.call("f", {refQ0, refQ1});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief Two call sites that cancel the same gate share a single commuted copy
 * of the callee.
 */
TEST_F(QCOQuantumIPOTest, reuseBoundaryCommutationAcrossCallSites) {
  const auto qubitType = programBuilder.getQubitType();

  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {qubitType}, {qubitType});
  programBuilder.endFunction({programBuilder.h(programBuilder.x(args[0]))});

  auto q0 = programBuilder.x(programBuilder.allocQubit());
  auto q1 = programBuilder.x(programBuilder.allocQubit());
  auto results0 = programBuilder.call("f", {q0});
  auto results1 = programBuilder.call("f", {q1});
  programBuilder.sink(results0[0]);
  programBuilder.sink(results1[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs = referenceBuilder.startFunction("f", {qubitType}, {qubitType});
  referenceBuilder.endFunction(
      {referenceBuilder.h(referenceBuilder.x(refArgs[0]))});
  auto specArgs = referenceBuilder.startFunction("f_spec_boundary_commutation",
                                                 {qubitType}, {qubitType});
  referenceBuilder.endFunction({referenceBuilder.h(specArgs[0])});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refResults0 =
      referenceBuilder.call("f_spec_boundary_commutation", {refQ0});
  auto refResults1 =
      referenceBuilder.call("f_spec_boundary_commutation", {refQ1});
  referenceBuilder.sink(refResults0[0]);
  referenceBuilder.sink(refResults1[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

// ==========================================================================
// Integration tests combining several IPO approaches.
// ==========================================================================

/**
 * @brief A callee that both starts with a gate that is trivial on |0> and uses
 * an internal auxiliary qubit is first specialized and then hoisted. The
 * hoisting applies to the original and the specialized copy alike.
 */
TEST_F(QCOQuantumIPOTest, specializationAndHoistingCombined) {
  const auto qubitType = programBuilder.getQubitType();

  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {qubitType}, {qubitType});
  auto aux = programBuilder.allocQubit();
  auto target = programBuilder.z(args[0]);
  std::tie(aux, target) = programBuilder.cx(aux, target);
  programBuilder.sink(aux);
  programBuilder.endFunction({target});

  auto q = programBuilder.allocQubit();
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  // The original keeps its `z` gate, but is hoisted as well.
  auto refArgs = referenceBuilder.startFunction("f", {qubitType, qubitType},
                                                {qubitType, qubitType});
  auto refAux = refArgs[1];
  auto refTarget = referenceBuilder.z(refArgs[0]);
  std::tie(refAux, refTarget) = referenceBuilder.cx(refAux, refTarget);
  refAux = referenceBuilder.reset(refAux);
  referenceBuilder.endFunction({refTarget, refAux});

  // The specialization drops the `z` gate and is hoisted too.
  auto specArgs = referenceBuilder.startFunction(
      "f_spec_zero_arg_0", {qubitType, qubitType}, {qubitType, qubitType});
  auto specAux = specArgs[1];
  auto specTarget = specArgs[0];
  std::tie(specAux, specTarget) = referenceBuilder.cx(specAux, specTarget);
  specAux = referenceBuilder.reset(specAux);
  referenceBuilder.endFunction({specTarget, specAux});

  auto refQ = referenceBuilder.allocQubit();
  auto refAuxAlloc = referenceBuilder.allocQubit();
  auto refResults =
      referenceBuilder.call("f_spec_zero_arg_0", {refQ, refAuxAlloc});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief A callee with two qubit arguments where one argument is specialized
 * for the |0> state and the other cancels a gate across the call boundary.
 */
TEST_F(QCOQuantumIPOTest, specializationAndBoundaryCommutationCombined) {
  const auto qubitType = programBuilder.getQubitType();

  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {qubitType, qubitType},
                                           {qubitType, qubitType});
  auto first = programBuilder.z(args[0]);
  auto second = programBuilder.x(args[1]);
  second = programBuilder.h(second);
  programBuilder.endFunction({first, second});

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.x(programBuilder.allocQubit());
  auto results = programBuilder.call("f", {q0, q1});
  programBuilder.sink(results[0]);
  programBuilder.sink(results[1]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs = referenceBuilder.startFunction("f", {qubitType, qubitType},
                                                {qubitType, qubitType});
  auto refFirst = referenceBuilder.z(refArgs[0]);
  auto refSecond = referenceBuilder.x(refArgs[1]);
  refSecond = referenceBuilder.h(refSecond);
  referenceBuilder.endFunction({refFirst, refSecond});

  // The |0> specialization drops the `z` gate on the first argument.
  auto specArgs = referenceBuilder.startFunction(
      "f_spec_zero_arg_0", {qubitType, qubitType}, {qubitType, qubitType});
  auto specSecond = referenceBuilder.x(specArgs[1]);
  specSecond = referenceBuilder.h(specSecond);
  referenceBuilder.endFunction({specArgs[0], specSecond});

  // Boundary commutation then removes the `x` gates around the call.
  auto commutedArgs = referenceBuilder.startFunction(
      "f_spec_zero_arg_0_spec_boundary_commutation", {qubitType, qubitType},
      {qubitType, qubitType});
  referenceBuilder.endFunction(
      {commutedArgs[0], referenceBuilder.h(commutedArgs[1])});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refResults = referenceBuilder.call(
      "f_spec_zero_arg_0_spec_boundary_commutation", {refQ0, refQ1});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}

/**
 * @brief A program with several distinct callees, each hitting a different IPO
 * approach: a |0> specialization, a fixed rotation angle, and a cancellation
 * across the call boundary.
 */
TEST_F(QCOQuantumIPOTest, multipleFunctionsWithDistinctOptimizations) {
  const auto qubitType = programBuilder.getQubitType();
  const auto floatType = programBuilder.getF64Type();

  programBuilder.initialize();
  auto prepareArgs =
      programBuilder.startFunction("prepare", {qubitType}, {qubitType});
  programBuilder.endFunction(
      {programBuilder.h(programBuilder.z(prepareArgs[0]))});

  auto rotateArgs = programBuilder.startFunction(
      "rotate", {qubitType, floatType}, {qubitType});
  programBuilder.endFunction({programBuilder.rz(rotateArgs[1], rotateArgs[0])});

  auto flipArgs =
      programBuilder.startFunction("flip", {qubitType}, {qubitType});
  programBuilder.endFunction({programBuilder.y(programBuilder.x(flipArgs[0]))});

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.x(programBuilder.allocQubit());
  auto prepared = programBuilder.call("prepare", {q0});
  auto angle = programBuilder.floatConstant(std::numbers::pi / 2);
  auto rotated = programBuilder.call("rotate", {q1, angle});
  auto flipped = programBuilder.call("flip", {q2});
  programBuilder.sink(prepared[0]);
  programBuilder.sink(rotated[0]);
  programBuilder.sink(flipped[0]);
  module = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refPrepareArgs =
      referenceBuilder.startFunction("prepare", {qubitType}, {qubitType});
  referenceBuilder.endFunction(
      {referenceBuilder.h(referenceBuilder.z(refPrepareArgs[0]))});
  auto preparedSpecArgs = referenceBuilder.startFunction(
      "prepare_spec_zero_arg_0", {qubitType}, {qubitType});
  referenceBuilder.endFunction({referenceBuilder.h(preparedSpecArgs[0])});

  auto refRotateArgs = referenceBuilder.startFunction(
      "rotate", {qubitType, floatType}, {qubitType});
  referenceBuilder.endFunction(
      {referenceBuilder.rz(refRotateArgs[1], refRotateArgs[0])});
  auto rotateSpecArgs = referenceBuilder.startFunction(
      "rotate_spec_fixed_angle_1", {qubitType, floatType}, {qubitType});
  referenceBuilder.endFunction(
      {referenceBuilder.rz(std::numbers::pi / 2, rotateSpecArgs[0])});

  auto refFlipArgs =
      referenceBuilder.startFunction("flip", {qubitType}, {qubitType});
  referenceBuilder.endFunction(
      {referenceBuilder.y(referenceBuilder.x(refFlipArgs[0]))});
  auto flipSpecArgs = referenceBuilder.startFunction(
      "flip_spec_boundary_commutation", {qubitType}, {qubitType});
  referenceBuilder.endFunction({referenceBuilder.y(flipSpecArgs[0])});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refQ2 = referenceBuilder.allocQubit();
  auto refPrepared = referenceBuilder.call("prepare_spec_zero_arg_0", {refQ0});
  auto refAngle = referenceBuilder.floatConstant(std::numbers::pi / 2);
  auto refRotated =
      referenceBuilder.call("rotate_spec_fixed_angle_1", {refQ1, refAngle});
  auto refFlipped =
      referenceBuilder.call("flip_spec_boundary_commutation", {refQ2});
  referenceBuilder.sink(refPrepared[0]);
  referenceBuilder.sink(refRotated[0]);
  referenceBuilder.sink(refFlipped[0]);
  reference = referenceBuilder.finalize();

  expectModuleMatchesReference();
}
