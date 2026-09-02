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
 * @file test_qco_auxiliary_qubit_hoisting.cpp
 * @brief Tests for the `quantum-auxiliary-qubit-hoisting` pass.
 */

#include "IPOTestFixture.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <tuple>
#include <utility>

namespace {

using QCOAuxiliaryQubitHoistingTest = ::mqt::test::IPOTestBase;
using namespace mlir;
using namespace mlir::qco;

// Auxiliary qubit hoisting.
// ==========================================================================

/**
 * @brief A qubit that a callee allocates and releases internally is turned into
 * an extra argument, so the caller owns the allocation and can reuse it.
 */
TEST_F(QCOAuxiliaryQubitHoistingTest, hoistAuxiliaryQubitIntoCaller) {
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
  moduleOp = programBuilder.finalize();

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

  expectSingleStageMatchesReference(createAuxiliaryQubitHoisting());
}

/**
 * @brief A qubit that the callee allocates but hands back to the caller is not
 * auxiliary and must stay where it is.
 */
TEST_F(QCOAuxiliaryQubitHoistingTest, noHoistingForReturnedQubit) {
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
  moduleOp = programBuilder.finalize();

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

  expectSingleStageMatchesReference(createAuxiliaryQubitHoisting());
}

/**
 * @brief The auxiliary qubit is tracked across a measurement and a reset on its
 * way to the release point.
 *
 * The measurement outcome is handed back to the caller so that the measurement
 * is not dead, and the reset sits between two gates so that it is neither
 * folded into the allocation nor into the release.
 */
TEST_F(QCOAuxiliaryQubitHoistingTest,
       hoistAuxiliaryQubitThroughMeasureAndReset) {
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
  moduleOp = programBuilder.finalize({results[1]});

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

  expectSingleStageMatchesReference(createAuxiliaryQubitHoisting());
}

/**
 * @brief The auxiliary qubit is tracked while it is parked in a tensor, past
 * an extraction of an unrelated element.
 */
TEST_F(QCOAuxiliaryQubitHoistingTest, hoistAuxiliaryQubitThroughTensor) {
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
  moduleOp = programBuilder.finalize();

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

  expectSingleStageMatchesReference(createAuxiliaryQubitHoisting());
}

/**
 * @brief The auxiliary qubit is tracked across a nested call on its way to the
 * release point.
 *
 * The nested callee returns more than one qubit and the auxiliary one is not
 * the first, so the walk has to match the operand position rather than simply
 * taking the first result.
 */
TEST_F(QCOAuxiliaryQubitHoistingTest, hoistAuxiliaryQubitThroughNestedCall) {
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
  moduleOp = programBuilder.finalize();

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

  expectSingleStageMatchesReference(createAuxiliaryQubitHoisting());
}

/**
 * @brief Every call site of a hoisted callee gets its own allocation.
 */
TEST_F(QCOAuxiliaryQubitHoistingTest,
       hoistAuxiliaryQubitWithMultipleCallSites) {
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
  moduleOp = programBuilder.finalize();

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

  expectSingleStageMatchesReference(createAuxiliaryQubitHoisting());
}

/**
 * @brief An allocation threaded through a recursive callee is not hoisted.
 */
TEST_F(QCOAuxiliaryQubitHoistingTest, noHoistingForRecursiveFunction) {
  auto module = parseModule(R"mlir(
func.func private @recursive(%q: !qco.qubit) -> !qco.qubit {
  %r = func.call @recursive(%q) : (!qco.qubit) -> !qco.qubit
  return %r : !qco.qubit
}
func.func private @outer(%q: !qco.qubit) -> !qco.qubit {
  %aux = qco.alloc : !qco.qubit
  %r = func.call @recursive(%aux) : (!qco.qubit) -> !qco.qubit
  qco.sink %r : !qco.qubit
  return %q : !qco.qubit
}
func.func @main(%q: !qco.qubit) -> !qco.qubit {
  %r = func.call @outer(%q) : (!qco.qubit) -> !qco.qubit
  return %r : !qco.qubit
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(
      runStage(module.get(), createAuxiliaryQubitHoisting()).succeeded());
  EXPECT_EQ(countAllocsIn(module.get(), "outer"), 1U);
}

/**
 * @brief An allocation nested inside a region is not hoisted, because it is not
 * executed on every path through the function.
 */
TEST_F(QCOAuxiliaryQubitHoistingTest, noHoistingForAllocInsideRegion) {
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
  moduleOp = programBuilder.finalize();

  buildProgram(referenceBuilder);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createAuxiliaryQubitHoisting());
}

/**
 * @brief The auxiliary qubit is tracked when it enters a tensor through an
 * insertion and while unrelated elements are moved in and out around it.
 */
TEST_F(QCOAuxiliaryQubitHoistingTest, hoistAuxiliaryQubitParkedInTensor) {
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
  moduleOp = programBuilder.finalize();

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

  expectSingleStageMatchesReference(createAuxiliaryQubitHoisting());
}

/**
 * @brief A call that only consumes linear values, and one that only produces
 * them, keep the builder's tracking consistent.
 */
TEST_F(QCOAuxiliaryQubitHoistingTest, callConsumesAndProducesLinearValues) {
  const auto qubitType = programBuilder.getQubitType();
  const auto tensorType = programBuilder.getQubitTensorType(2);

  const auto buildProgram = [&](QCOProgramBuilder& b) {
    b.initialize();
    auto consumeArgs = b.startFunction("consume", {qubitType, tensorType}, {});
    b.sink(consumeArgs[0]);
    b.qtensorDealloc(consumeArgs[1]);
    b.endFunction({});

    b.startFunction("produce", {}, {tensorType});
    b.endFunction({b.qtensorAlloc(2)});

    auto q = b.allocQubit();
    auto scratch = b.qtensorAlloc(2);
    b.call("consume", {q, scratch});
    auto produced = b.call("produce", {});
    b.qtensorDealloc(produced[0]);
  };

  buildProgram(programBuilder);
  moduleOp = programBuilder.finalize();
  buildProgram(referenceBuilder);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createAuxiliaryQubitHoisting());
}

/**
 * @brief The auxiliary qubit is tracked across a call whose callee also returns
 * a classical value ahead of the qubit.
 */
TEST_F(QCOAuxiliaryQubitHoistingTest,
       hoistAuxiliaryQubitThroughCallWithClassicalResult) {
  const auto qubitType = programBuilder.getQubitType();
  const auto bitType = programBuilder.getI1Type();

  const auto buildCallee = [&](QCOProgramBuilder& b) {
    auto innerArgs = b.startFunction("g", {qubitType}, {bitType, qubitType});
    auto [inner, bit] = b.measure(innerArgs[0]);
    b.endFunction({bit, inner});
  };

  programBuilder.initialize({bitType});
  buildCallee(programBuilder);

  auto args =
      programBuilder.startFunction("f", {qubitType}, {qubitType, bitType});
  auto nested = programBuilder.call("g", {programBuilder.allocQubit()});
  auto aux = nested[1];
  auto target = args[0];
  std::tie(aux, target) = programBuilder.cx(aux, target);
  programBuilder.sink(aux);
  programBuilder.endFunction({target, nested[0]});

  auto q = programBuilder.allocQubit();
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize({results[1]});

  referenceBuilder.initialize({bitType});
  buildCallee(referenceBuilder);

  auto refArgs = referenceBuilder.startFunction(
      "f", {qubitType, qubitType}, {qubitType, bitType, qubitType});
  auto refNested = referenceBuilder.call("g", {refArgs[1]});
  auto refAux = refNested[1];
  auto refTarget = refArgs[0];
  std::tie(refAux, refTarget) = referenceBuilder.cx(refAux, refTarget);
  refAux = referenceBuilder.reset(refAux);
  referenceBuilder.endFunction({refTarget, refNested[0], refAux});

  auto refQ = referenceBuilder.allocQubit();
  auto refAuxAlloc = referenceBuilder.allocQubit();
  auto refResults = referenceBuilder.call("f", {refQ, refAuxAlloc});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[2]);
  reference = referenceBuilder.finalize({refResults[1]});

  expectSingleStageMatchesReference(createAuxiliaryQubitHoisting());
}

/**
 * @brief A qubit handed to a callee that keeps it is not auxiliary, so it is
 * left alone rather than indexed past the end of the call's results.
 */
TEST_F(QCOAuxiliaryQubitHoistingTest, noHoistingWhenCalleeKeepsAuxiliaryQubit) {
  const auto qubitType = programBuilder.getQubitType();

  const auto buildProgram = [&](QCOProgramBuilder& b) {
    b.initialize();
    auto consumeArgs = b.startFunction("consume", {qubitType}, {});
    b.sink(consumeArgs[0]);
    b.endFunction({});

    auto args = b.startFunction("f", {qubitType}, {qubitType});
    b.call("consume", {b.allocQubit()});
    b.endFunction({b.h(args[0])});

    auto q = b.allocQubit();
    auto results = b.call("f", {q});
    b.sink(results[0]);
  };

  buildProgram(programBuilder);
  moduleOp = programBuilder.finalize();
  buildProgram(referenceBuilder);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createAuxiliaryQubitHoisting());
}

/**
 * @brief Hoisting reaches the outermost caller regardless of declaration order.
 *
 * @details
 * An allocation hoisted into a caller may be hoistable again. Processing in
 * module order used to strand it wherever the declarations happened to sit.
 */
TEST_F(QCOAuxiliaryQubitHoistingTest, hoistingIsIndependentOfDeclarationOrder) {
  const auto qubitType = programBuilder.getQubitType();

  // Builds `main -> mid -> leaf`, where `leaf` owns an auxiliary qubit.
  const auto build = [&](QCOProgramBuilder& builder) {
    builder.initialize();
    auto leafArgs = builder.startFunction("leaf", {qubitType}, {qubitType});
    auto aux = builder.allocQubit();
    auto target = leafArgs[0];
    std::tie(aux, target) = builder.cx(aux, target);
    builder.sink(aux);
    builder.endFunction({target});

    auto midArgs = builder.startFunction("mid", {qubitType}, {qubitType});
    auto midResults = builder.call("leaf", {midArgs[0]});
    builder.endFunction({midResults[0]});

    auto q = builder.allocQubit();
    auto results = builder.call("mid", {q});
    builder.sink(results[0]);
    return builder.finalize();
  };

  moduleOp = build(programBuilder);
  reference = build(referenceBuilder);

  // The builder has to declare a callee before the call, so the second module
  // is reordered afterwards. Both now describe the same call graph and differ
  // only in the order the module walk visits the two callees.
  auto refLeaf = reference->lookupSymbol<func::FuncOp>("leaf");
  auto refMid = reference->lookupSymbol<func::FuncOp>("mid");
  ASSERT_TRUE(refLeaf);
  ASSERT_TRUE(refMid);
  refLeaf->moveAfter(refMid.getOperation());

  ASSERT_TRUE(
      runStage(moduleOp.get(), createAuxiliaryQubitHoisting()).succeeded());
  ASSERT_TRUE(
      runStage(reference.get(), createAuxiliaryQubitHoisting()).succeeded());

  // The auxiliary allocation belongs in the entry function either way: one
  // allocation for the qubit passed in and one for the hoisted auxiliary.
  for (auto* module : {&moduleOp, &reference}) {
    EXPECT_EQ(countAllocsIn(module->get(), "leaf"), 0U);
    EXPECT_EQ(countAllocsIn(module->get(), "mid"), 0U);
    EXPECT_EQ(countAllocsIn(module->get(), "main"), 2U);
  }
}

// ==========================================================================

} // namespace
