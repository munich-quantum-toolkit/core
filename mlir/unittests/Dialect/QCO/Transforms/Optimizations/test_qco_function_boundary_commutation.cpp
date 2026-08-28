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
 * @file test_qco_function_boundary_commutation.cpp
 * @brief Tests for the `quantum-function-boundary-commutation` pass.
 */

#include "IPOTestFixture.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <gtest/gtest.h>
#include <mlir/Support/LLVM.h>

#include <tuple>

namespace {

using QCOFunctionBoundaryCommutationTest = ::mqt::test::IPOTestBase;
using namespace mlir;
using namespace mlir::qco;

// Quantum function boundary commutation.
// ==========================================================================

/**
 * @brief A self-inverse gate applied right before a call cancels with the same
 * gate at the start of the callee.
 */
TEST_F(QCOFunctionBoundaryCommutationTest,
       cancelSelfInverseGateAcrossCallBoundary) {
  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {programBuilder.getQubitType()},
                                           {programBuilder.getQubitType()});
  programBuilder.endFunction({programBuilder.h(programBuilder.x(args[0]))});

  auto q = programBuilder.x(programBuilder.allocQubit());
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();

  // Both the caller-side and the callee-side gate disappear.
  auto specArgs = referenceBuilder.startFunction(
      "f_spec_boundary_commutation_arg_0", {referenceBuilder.getQubitType()},
      {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction({referenceBuilder.h(specArgs[0])});

  auto refQ = referenceBuilder.allocQubit();
  auto refResults =
      referenceBuilder.call("f_spec_boundary_commutation_arg_0", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumFunctionBoundaryCommutation());
}

/**
 * @brief Two different gates across the call boundary do not cancel.
 */
TEST_F(QCOFunctionBoundaryCommutationTest, noCancellationForDifferentGates) {
  programBuilder.initialize();
  auto args = programBuilder.startFunction("f", {programBuilder.getQubitType()},
                                           {programBuilder.getQubitType()});
  programBuilder.endFunction({programBuilder.y(args[0])});

  auto q = programBuilder.x(programBuilder.allocQubit());
  auto results = programBuilder.call("f", {q});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  auto refArgs =
      referenceBuilder.startFunction("f", {referenceBuilder.getQubitType()},
                                     {referenceBuilder.getQubitType()});
  referenceBuilder.endFunction({referenceBuilder.y(refArgs[0])});

  auto refQ = referenceBuilder.x(referenceBuilder.allocQubit());
  auto refResults = referenceBuilder.call("f", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumFunctionBoundaryCommutation());
}

/**
 * @brief Controlled gates are out of scope for boundary commutation, even when
 * the same one appears on both sides of the call. Cancelling them would require
 * reasoning about the control qubits as well.
 */
TEST_F(QCOFunctionBoundaryCommutationTest, noCancellationForControlledGates) {
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
  moduleOp = programBuilder.finalize();

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

  expectSingleStageMatchesReference(createQuantumFunctionBoundaryCommutation());
}

/**
 * @brief Two call sites that cancel the same gate share a single commuted copy
 * of the callee.
 */
TEST_F(QCOFunctionBoundaryCommutationTest,
       reuseBoundaryCommutationAcrossCallSites) {
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
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  auto specArgs = referenceBuilder.startFunction(
      "f_spec_boundary_commutation_arg_0", {qubitType}, {qubitType});
  referenceBuilder.endFunction({referenceBuilder.h(specArgs[0])});

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refResults0 =
      referenceBuilder.call("f_spec_boundary_commutation_arg_0", {refQ0});
  auto refResults1 =
      referenceBuilder.call("f_spec_boundary_commutation_arg_0", {refQ1});
  referenceBuilder.sink(refResults0[0]);
  referenceBuilder.sink(refResults1[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumFunctionBoundaryCommutation());
}

/**
 * @brief Two call sites that cancel a gate on different parameters of the same
 * callee must get their own specialization, because the gate is removed from a
 * specific argument.
 */
TEST_F(QCOFunctionBoundaryCommutationTest,
       separateCommutationSpecializationPerParameter) {
  const auto qubitType = programBuilder.getQubitType();

  const auto buildCallee = [&qubitType](QCOProgramBuilder& b, StringRef name) {
    auto args =
        b.startFunction(name, {qubitType, qubitType}, {qubitType, qubitType});
    b.endFunction({b.x(args[0]), b.x(args[1])});
  };

  programBuilder.initialize();
  buildCallee(programBuilder, "f");

  // The first call cancels the gate on parameter 0, ...
  auto a0 = programBuilder.x(programBuilder.allocQubit());
  auto a1 = programBuilder.allocQubit();
  auto results0 = programBuilder.call("f", {a0, a1});
  // ... the second one on parameter 1.
  auto b0 = programBuilder.allocQubit();
  auto b1 = programBuilder.x(programBuilder.allocQubit());
  auto results1 = programBuilder.call("f", {b0, b1});
  programBuilder.sink(results0[0]);
  programBuilder.sink(results0[1]);
  programBuilder.sink(results1[0]);
  programBuilder.sink(results1[1]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  // Both call sites are redirected, so the original is left without callers.
  // One specialization without the gate on parameter 0, ...
  auto spec0Args = referenceBuilder.startFunction(
      "f_spec_boundary_commutation_arg_0", {qubitType, qubitType},
      {qubitType, qubitType});
  referenceBuilder.endFunction(
      {spec0Args[0], referenceBuilder.x(spec0Args[1])});
  // ... and one without the gate on parameter 1.
  auto spec1Args = referenceBuilder.startFunction(
      "f_spec_boundary_commutation_arg_1", {qubitType, qubitType},
      {qubitType, qubitType});
  referenceBuilder.endFunction(
      {referenceBuilder.x(spec1Args[0]), spec1Args[1]});

  auto refA0 = referenceBuilder.allocQubit();
  auto refA1 = referenceBuilder.allocQubit();
  auto refResults0 = referenceBuilder.call("f_spec_boundary_commutation_arg_0",
                                           {refA0, refA1});
  auto refB0 = referenceBuilder.allocQubit();
  auto refB1 = referenceBuilder.allocQubit();
  auto refResults1 = referenceBuilder.call("f_spec_boundary_commutation_arg_1",
                                           {refB0, refB1});
  referenceBuilder.sink(refResults0[0]);
  referenceBuilder.sink(refResults0[1]);
  referenceBuilder.sink(refResults1[0]);
  referenceBuilder.sink(refResults1[1]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumFunctionBoundaryCommutation());
}

// ==========================================================================

} // namespace
