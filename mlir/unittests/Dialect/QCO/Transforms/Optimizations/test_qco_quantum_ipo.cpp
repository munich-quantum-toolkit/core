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
 * @file test_qco_quantum_ipo.cpp
 * @brief Cross-stage tests for the `quantum-ipo` pipeline.
 *
 * @details
 * Only scenarios that need more than one pass belong here. Everything about a
 * single pass lives in that pass's own suite, scheduled on the pass alone.
 */

#include "IPOTestFixture.h"
#include "Support/IRVerification.h"
#include "mlir/Support/Passes.h"

#include <gtest/gtest.h>
#include <mlir/Support/LLVM.h>

#include <numbers>
#include <tuple>

namespace {

using QCOQuantumIPOPipelineTest = ::mqt::test::IPOTestBase;
using namespace mlir;
using namespace mlir::qco;

// Integration tests combining several IPO approaches.
// ==========================================================================

/**
 * @brief A callee that both starts with a gate that is trivial on |0> and uses
 * an internal auxiliary qubit is first specialized and then hoisted. The
 * hoisting applies to the original and the specialized copy alike.
 */
TEST_F(QCOQuantumIPOPipelineTest, specializationAndHoistingCombined) {
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
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  // The original loses its only caller to the specialization and is dropped
  // before hoisting runs, so it is never given an auxiliary argument.
  // The specialization drops the `z` gate and is hoisted.
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

  expectPipelineMatchesReference();
}

/**
 * @brief A callee with two qubit arguments where one argument is specialized
 * for the |0> state and the other cancels a gate across the call boundary.
 */
TEST_F(QCOQuantumIPOPipelineTest,
       specializationAndBoundaryCommutationCombined) {
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
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  // The |0> specialization drops the `z` gate on the first argument and the
  // boundary commutation then specializes that copy in turn, so both the
  // original and the intermediate end up without callers.
  // Only the last link of the chain survives: it has neither the `z` gate on
  // the first argument nor the `x` gate on the second.
  auto commutedArgs = referenceBuilder.startFunction(
      "f_spec_zero_arg_0_spec_boundary_commutation_arg_1",
      {qubitType, qubitType}, {qubitType, qubitType});
  referenceBuilder.endFunction(
      {commutedArgs[0], referenceBuilder.h(commutedArgs[1])});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refResults = referenceBuilder.call(
      "f_spec_zero_arg_0_spec_boundary_commutation_arg_1", {refQ0, refQ1});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectPipelineMatchesReference();
}

/**
 * @brief A program with several distinct callees, each hitting a different IPO
 * approach: a |0> specialization, a fixed rotation angle, and a cancellation
 * across the call boundary.
 */
TEST_F(QCOQuantumIPOPipelineTest, multipleFunctionsWithDistinctOptimizations) {
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
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  // Each original loses its only caller to its specialization and is dropped.
  auto preparedSpecArgs = referenceBuilder.startFunction(
      "prepare_spec_zero_arg_0", {qubitType}, {qubitType});
  referenceBuilder.endFunction({referenceBuilder.h(preparedSpecArgs[0])});

  auto rotateSpecArgs = referenceBuilder.startFunction(
      "rotate_spec_fixed_angle_1", {qubitType, floatType}, {qubitType});
  referenceBuilder.endFunction(
      {referenceBuilder.rz(std::numbers::pi / 2, rotateSpecArgs[0])});

  auto flipSpecArgs = referenceBuilder.startFunction(
      "flip_spec_boundary_commutation_arg_0", {qubitType}, {qubitType});
  referenceBuilder.endFunction({referenceBuilder.y(flipSpecArgs[0])});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refQ2 = referenceBuilder.allocQubit();
  auto refPrepared = referenceBuilder.call("prepare_spec_zero_arg_0", {refQ0});
  auto refAngle = referenceBuilder.floatConstant(std::numbers::pi / 2);
  auto refRotated =
      referenceBuilder.call("rotate_spec_fixed_angle_1", {refQ1, refAngle});
  auto refFlipped =
      referenceBuilder.call("flip_spec_boundary_commutation_arg_0", {refQ2});
  referenceBuilder.sink(refPrepared[0]);
  referenceBuilder.sink(refRotated[0]);
  referenceBuilder.sink(refFlipped[0]);
  reference = referenceBuilder.finalize();

  expectPipelineMatchesReference();
}

// ==========================================================================

} // namespace

/**
 * @brief The pipeline is reachable under the name it registers.
 *
 * @details
 * Scheduling the passes directly, as the cases above do, would still pass if
 * `quantum-ipo` were never registered. Running it by name is what `mqt-cc`
 * does, so this is the contract that has to hold for the pipeline to be usable
 * from the command line at all.
 */
TEST_F(QCOQuantumIPOPipelineTest, PipelineRunsUnderItsRegisteredName) {
  const auto qubitType = programBuilder.getQubitType();

  const auto build = [&qubitType](QCOProgramBuilder& b) {
    b.initialize();
    auto args = b.startFunction("f", {qubitType}, {qubitType});
    b.endFunction({b.z(args[0])});
    auto q = b.allocQubit();
    auto results = b.call("f", {q});
    b.sink(results[0]);
    return b.finalize();
  };

  moduleOp = build(programBuilder);
  ASSERT_TRUE(
      runPassPipeline(moduleOp.get(), "quantum-ipo", false, false).succeeded());

  // The `z` acts trivially on |0>, so the specialized callee drops it.
  reference = build(referenceBuilder);
  ASSERT_TRUE(runQuantumIPOPipeline(reference.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(moduleOp.get()).succeeded());
  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(moduleOp.get(), reference.get()));
}
