/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/**
 * @file test_qco_quantum_argument_promotion.cpp
 * @brief Tests for the `quantum-argument-promotion` pass.
 */

#include "IPOTestFixture.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>

#include <tuple>

namespace {

using QCOQuantumArgumentPromotionTest = ::mqt::test::IPOTestBase;
using namespace mlir;
using namespace mlir::qco;

// Quantum argument promotion.
// ==========================================================================

/**
 * @brief A tensor argument whose elements are extracted and re-inserted at
 * compile-time constant indices is replaced by scalar qubit arguments.
 */
TEST_F(QCOQuantumArgumentPromotionTest, promoteTensorArgumentToQubitArgument) {
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
  moduleOp = programBuilder.finalize();

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

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/**
 * @brief An element that is taken out and put straight back at the same index,
 * without any gate in between, leaves nothing to promote once it is folded.
 *
 * @details
 * The folder collapses such an extract/insert pair back into the original
 * tensor, which leaves the callee as an identity function. That fold is the
 * precondition here, so it is applied explicitly: this pass does not run a
 * folder of its own, and promoting an unfolded pass-through is wasted work
 * rather than a miscompile.
 */
TEST_F(QCOQuantumArgumentPromotionTest, noPromotionForFoldedPassThrough) {
  const auto tensorType = programBuilder.getQubitTensorType(2);

  const auto buildProgram = [&tensorType](QCOProgramBuilder& b) {
    b.initialize();
    auto args = b.startFunction("f", {tensorType}, {tensorType});
    auto [rest, inner] = b.qtensorExtract(args[0], 0);
    b.endFunction({b.qtensorInsert(inner, rest, 0)});

    auto q0 = b.allocQubit();
    auto q1 = b.allocQubit();
    auto tensor = b.qtensorFromElements({q0, q1});
    auto results = b.call("f", {tensor});
    b.qtensorDealloc(results[0]);
  };

  buildProgram(programBuilder);
  moduleOp = programBuilder.finalize();
  buildProgram(referenceBuilder);
  reference = referenceBuilder.finalize();

  ASSERT_TRUE(runCanonicalizerPass(moduleOp.get()).succeeded());
  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/**
 * @brief Only the tensor elements the callee actually touches become scalar
 * arguments; untouched elements never cross the call boundary.
 */
TEST_F(QCOQuantumArgumentPromotionTest, promoteOnlyUsedTensorElements) {
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
  moduleOp = programBuilder.finalize();

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

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/**
 * @brief A qubit that is moved to a different slot is promoted with the
 * extraction and insertion indices kept apart.
 */
TEST_F(QCOQuantumArgumentPromotionTest, promoteTensorElementIntoDifferentSlot) {
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
  moduleOp = programBuilder.finalize();

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

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/**
 * @brief An element that is extracted but never re-inserted cannot be promoted,
 * because the promoted callee would have nothing to hand back for that slot.
 */
TEST_F(QCOQuantumArgumentPromotionTest, noPromotionWithoutMatchingInsert) {
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
  moduleOp = programBuilder.finalize();

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

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/**
 * @brief An element whose path from extraction to re-insertion runs through a
 * call is not promoted.
 *
 * @details
 * The walk only recognises the operations it knows to thread a qubit. A call is
 * not one of them, and guessing that its first result carries the qubit on
 * would let a slot be promoted that no longer holds the extracted qubit, so the
 * callee is left alone.
 */
TEST_F(QCOQuantumArgumentPromotionTest,
       noPromotionWhenCallSitsOnExtractedPath) {
  const auto tensorType = programBuilder.getQubitTensorType(2);

  const auto buildProgram = [&](QCOProgramBuilder& b) {
    const auto qubitType = b.getQubitType();
    b.initialize();

    // The helper takes the qubit as its second operand and returns one qubit.
    auto helperArgs =
        b.startFunction("helper", {qubitType, qubitType}, {qubitType});
    b.sink(helperArgs[0]);
    b.endFunction({b.h(helperArgs[1])});

    auto args = b.startFunction("f", {tensorType}, {tensorType});
    auto [rest, inner] = b.qtensorExtract(args[0], 0);
    auto helped = b.call("helper", {b.allocQubit(), inner})[0];
    b.endFunction({b.qtensorInsert(helped, rest, 0)});

    auto q0 = b.allocQubit();
    auto q1 = b.allocQubit();
    auto tensor = b.qtensorFromElements({q0, q1});
    auto results = b.call("f", {tensor});
    b.qtensorDealloc(results[0]);
  };

  buildProgram(programBuilder);
  moduleOp = programBuilder.finalize();
  buildProgram(referenceBuilder);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/**
 * @brief A tensor argument that never has an element taken out of it has
 * nothing to promote.
 */
TEST_F(QCOQuantumArgumentPromotionTest, noPromotionWithoutElementAccess) {
  const auto tensorType = programBuilder.getQubitTensorType(2);

  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {tensorType}, {tensorType});
  programBuilder.endFunction({args[0]});

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto tensor = programBuilder.qtensorFromElements({q0, q1});
  auto results = programBuilder.call("f", {tensor});
  programBuilder.qtensorDealloc(results[0]);
  moduleOp = programBuilder.finalize();

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

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/**
 * @brief A callee that touches several tensor elements gets one scalar argument
 * and one scalar result per element.
 */
TEST_F(QCOQuantumArgumentPromotionTest, promoteMultipleTensorElements) {
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
  moduleOp = programBuilder.finalize();

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

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/**
 * @brief A promoted element may be measured inside the callee; the measurement
 * outcome stays a separate result and the caller keeps reading it.
 */
TEST_F(QCOQuantumArgumentPromotionTest, promoteTensorElementWithMeasurement) {
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
  moduleOp = programBuilder.finalize({results[1]});

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

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/**
 * @brief The tensor has to be handed back as the first result, because that is
 * the result the promoted qubits take the place of.
 */
TEST_F(QCOQuantumArgumentPromotionTest, noPromotionWhenTensorIsNotFirstResult) {
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

  moduleOp = programBuilder.finalize({buildProgram(programBuilder)});
  reference = referenceBuilder.finalize({buildProgram(referenceBuilder)});

  expectSingleStageMatchesReference(createQuantumArgumentPromotion());
}

/**
 * @brief A slot the callee writes before reading again must not be promoted.
 *
 * @details
 * Extractions move in front of the call and insertions behind it, so such a
 * read would be served from the caller's original tensor. Here the callee
 * computes `x(h(slot 0))`, which promotion would turn into `x(slot 1)`.
 */
TEST_F(QCOQuantumArgumentPromotionTest,
       noPromotionWhenSlotIsWrittenBeforeItIsRead) {
  auto module = parseModule(R"mlir(
func.func private @callee(%t: tensor<2x!qco.qubit>) -> tensor<2x!qco.qubit> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %t1, %q0 = qtensor.extract %t[%c0] : tensor<2x!qco.qubit>
  %q0h = qco.h %q0 : !qco.qubit -> !qco.qubit
  %t2 = qtensor.insert %q0h into %t1[%c1] : tensor<2x!qco.qubit>
  %t3, %q1 = qtensor.extract %t2[%c1] : tensor<2x!qco.qubit>
  %q1x = qco.x %q1 : !qco.qubit -> !qco.qubit
  %t4 = qtensor.insert %q1x into %t3[%c0] : tensor<2x!qco.qubit>
  return %t4 : tensor<2x!qco.qubit>
}
func.func @main(%t: tensor<2x!qco.qubit>) -> tensor<2x!qco.qubit> {
  %r = func.call @callee(%t) : (tensor<2x!qco.qubit>) -> tensor<2x!qco.qubit>
  return %r : tensor<2x!qco.qubit>
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(
      runStage(module.get(), createQuantumArgumentPromotion()).succeeded());

  auto callee = module->lookupSymbol<func::FuncOp>("callee");
  ASSERT_TRUE(callee);
  EXPECT_TRUE(isa<RankedTensorType>(callee.getArgumentTypes()[0]))
      << "the tensor argument must survive, the accesses depend on each other";
}

/**
 * @brief An insertion that does not belong to a promoted slot blocks promotion.
 *
 * @details
 * It survives the rewrite still using the tensor argument that is erased right
 * afterwards, which used to abort on MLIR's `use_empty()` assertion.
 */
TEST_F(QCOQuantumArgumentPromotionTest, noPromotionForUnmatchedInsertOnChain) {
  auto module = parseModule(R"mlir(
func.func private @callee(%t: tensor<2x!qco.qubit>, %extra: !qco.qubit) -> tensor<2x!qco.qubit> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %t1, %q0 = qtensor.extract %t[%c0] : tensor<2x!qco.qubit>
  %q0h = qco.h %q0 : !qco.qubit -> !qco.qubit
  %t2 = qtensor.insert %q0h into %t1[%c0] : tensor<2x!qco.qubit>
  %t3 = qtensor.insert %extra into %t2[%c1] : tensor<2x!qco.qubit>
  return %t3 : tensor<2x!qco.qubit>
}
func.func @main(%t: tensor<2x!qco.qubit>, %e: !qco.qubit) -> tensor<2x!qco.qubit> {
  %r = func.call @callee(%t, %e) : (tensor<2x!qco.qubit>, !qco.qubit) -> tensor<2x!qco.qubit>
  return %r : tensor<2x!qco.qubit>
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(
      runStage(module.get(), createQuantumArgumentPromotion()).succeeded());

  auto callee = module->lookupSymbol<func::FuncOp>("callee");
  ASSERT_TRUE(callee);
  EXPECT_TRUE(isa<RankedTensorType>(callee.getArgumentTypes()[0]))
      << "the tensor argument must survive, one insertion is unmatched";
}

// ==========================================================================

} // namespace
