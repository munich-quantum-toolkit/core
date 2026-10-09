/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file test_qco_quantum_ipo.cpp
/// Cross-stage tests for the `quantum-ipo` pipeline.
///
/// Only scenarios that need more than one pass belong here. Everything about a
/// single pass lives in that pass's own suite, scheduled on the pass alone.

#include "mqt/Support/Passes.h"

#include "IPOTestFixture.h"

#include "mlir/Support/LLVM.h"

#include <Support/IRVerification.h>
#include <gtest/gtest.h>
#include <numbers>
#include <tuple>

namespace {

using QCOQuantumIPOPipelineTest = ::mqt::test::IPOTestBase;
using namespace mlir;
using namespace mlir::qco;

// Integration tests combining several IPO approaches.
// ==========================================================================

/// A callee that starts with a gate that is trivial on |0> and uses an
/// auxiliary qubit is first specialized, and then the specialization is
/// hoisted. The original loses its only caller and is dropped.
TEST_F(QCOQuantumIPOPipelineTest, specializationAndHoistingCombined) {
  const auto qubitType = getQubitType();

  programBuilder.initialize();
  programBuilder.createFunction("f", {qubitType}, [&](ValueRange args) {
    auto [aux, target] = programBuilder.cx(programBuilder.allocQubit(),
                                           programBuilder.z(args[0]));
    programBuilder.sink(aux);
    return SmallVector<Value>{target};
  });

  auto q = programBuilder.allocQubit();
  auto results = callFunction(programBuilder, "f", {q});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction(
      "f_spec_zero_arg_0", {qubitType, qubitType}, [&](ValueRange specArgs) {
        auto [specAux, specTarget] =
            referenceBuilder.cx(specArgs[1], specArgs[0]);
        specAux = referenceBuilder.reset(specAux);
        return SmallVector<Value>{specTarget, specAux};
      });

  auto refQ = referenceBuilder.allocQubit();
  auto refAuxAlloc = referenceBuilder.allocQubit();
  auto refResults =
      callFunction(referenceBuilder, "f_spec_zero_arg_0", {refQ, refAuxAlloc});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectPipelineMatchesReference();
}

TEST_F(QCOQuantumIPOPipelineTest,
       specializationAndBoundaryCommutationCombined) {
  const auto qubitType = getQubitType();

  programBuilder.initialize();
  programBuilder.createFunction("f", {qubitType, qubitType},
                                [&](ValueRange args) {
                                  auto first = programBuilder.z(args[0]);
                                  auto second = programBuilder.x(args[1]);
                                  second = programBuilder.h(second);
                                  return SmallVector<Value>{first, second};
                                });

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.x(programBuilder.allocQubit());
  auto results = callFunction(programBuilder, "f", {q0, q1});
  programBuilder.sink(results[0]);
  programBuilder.sink(results[1]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  // The |0> specialization drops the `z` gate on the first argument and the
  // boundary commutation then specializes that copy in turn, so both the
  // original and the intermediate end up without callers.
  // Only the last link of the chain survives: it has neither the `z` gate on
  // the first argument nor the `x` gate on the second.
  referenceBuilder.createFunction(
      "f_spec_zero_arg_0_spec_boundary_commutation_arg_1",
      {qubitType, qubitType}, [&](ValueRange commutedArgs) {
        return SmallVector<Value>{
            commutedArgs[0],
            referenceBuilder.h(commutedArgs[1]),
        };
      });

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refResults = callFunction(
      referenceBuilder, "f_spec_zero_arg_0_spec_boundary_commutation_arg_1",
      {refQ0, refQ1});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectPipelineMatchesReference();
}

/// A program with several distinct callees, each hitting a different IPO
/// approach: a |0> specialization, a fixed rotation angle, and a cancellation
/// across the call boundary.
TEST_F(QCOQuantumIPOPipelineTest, multipleFunctionsWithDistinctOptimizations) {
  const auto qubitType = getQubitType();
  const auto floatType = programBuilder.getF64Type();

  programBuilder.initialize();
  programBuilder.createFunction(
      "prepare", {qubitType}, [&](ValueRange prepareArgs) {
        return SmallVector<Value>{
            programBuilder.h(programBuilder.z(prepareArgs[0])),
        };
      });

  programBuilder.createFunction(
      "rotate", {qubitType, floatType}, [&](ValueRange rotateArgs) {
        return SmallVector<Value>{
            programBuilder.rz(rotateArgs[1], rotateArgs[0]),
        };
      });

  programBuilder.createFunction("flip", {qubitType}, [&](ValueRange flipArgs) {
    return SmallVector<Value>{programBuilder.y(programBuilder.x(flipArgs[0]))};
  });

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto q2 = programBuilder.x(programBuilder.allocQubit());
  auto prepared = callFunction(programBuilder, "prepare", {q0});
  auto angle = programBuilder.floatConstant(std::numbers::pi / 2);
  auto rotated = callFunction(programBuilder, "rotate", {q1, angle});
  auto flipped = callFunction(programBuilder, "flip", {q2});
  programBuilder.sink(prepared[0]);
  programBuilder.sink(rotated[0]);
  programBuilder.sink(flipped[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  // Each original loses its only caller to its specialization and is dropped.
  referenceBuilder.createFunction(
      "prepare_spec_zero_arg_0", {qubitType}, [&](ValueRange preparedSpecArgs) {
        return SmallVector<Value>{referenceBuilder.h(preparedSpecArgs[0])};
      });

  referenceBuilder.createFunction(
      "rotate_spec_fixed_angle_1", {qubitType, floatType},
      [&](ValueRange rotateSpecArgs) {
        return SmallVector<Value>{
            referenceBuilder.rz(std::numbers::pi / 2, rotateSpecArgs[0]),
        };
      });

  referenceBuilder.createFunction("flip_spec_boundary_commutation_arg_0",
                                  {qubitType}, [&](ValueRange flipSpecArgs) {
                                    return SmallVector<Value>{
                                        referenceBuilder.y(flipSpecArgs[0]),
                                    };
                                  });

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refQ2 = referenceBuilder.allocQubit();
  auto refPrepared =
      callFunction(referenceBuilder, "prepare_spec_zero_arg_0", {refQ0});
  auto refAngle = referenceBuilder.floatConstant(std::numbers::pi / 2);
  auto refRotated = callFunction(referenceBuilder, "rotate_spec_fixed_angle_1",
                                 {refQ1, refAngle});
  auto refFlipped = callFunction(
      referenceBuilder, "flip_spec_boundary_commutation_arg_0", {refQ2});
  referenceBuilder.sink(refPrepared[0]);
  referenceBuilder.sink(refRotated[0]);
  referenceBuilder.sink(refFlipped[0]);
  reference = referenceBuilder.finalize();

  expectPipelineMatchesReference();
}

// ==========================================================================

} // namespace

/// The pipeline is reachable under the name it registers.
///
/// Scheduling the passes directly, as the cases above do, would still pass if
/// `quantum-ipo` were never registered. Running it by name is what `mqt-cc`
/// does, so this is the contract that has to hold for the pipeline to be usable
/// from the command line at all.
TEST_F(QCOQuantumIPOPipelineTest, PipelineRunsUnderItsRegisteredName) {
  const auto qubitType = getQubitType();

  const auto build = [&qubitType](QCOProgramBuilder& b) {
    b.initialize();
    b.createFunction("f", {qubitType}, [&](ValueRange args) {
      return SmallVector<Value>{b.z(args[0])};
    });
    auto q = b.allocQubit();
    auto results = callFunction(b, "f", {q});
    b.sink(results[0]);
    return b.finalize();
  };

  moduleOp = build(programBuilder);
  ASSERT_TRUE(runPassPipeline(moduleOp.get(), "quantum-ipo").succeeded());

  // The `z` acts trivially on |0>, so the specialized callee drops it.
  reference = build(referenceBuilder);
  ASSERT_TRUE(runQuantumIPOPipeline(reference.get()).succeeded());
  ASSERT_TRUE(runCanonicalizerPass(moduleOp.get()).succeeded());
  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(moduleOp.get(), reference.get()));
}
