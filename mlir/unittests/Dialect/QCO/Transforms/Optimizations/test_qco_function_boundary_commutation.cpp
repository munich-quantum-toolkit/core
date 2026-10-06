/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file test_qco_function_boundary_commutation.cpp
/// Tests for the `quantum-function-boundary-commutation` pass.

#include "mqt/Dialect/QCO/Transforms/Passes.h"

#include "IPOTestFixture.h"

#include "mlir/Support/LLVM.h"

#include <gtest/gtest.h>
#include <tuple>

namespace {

using QCOFunctionBoundaryCommutationTest = ::mqt::test::IPOTestBase;
using namespace mlir;
using namespace mlir::qco;

// Quantum function boundary commutation.
// ==========================================================================

/// A self-inverse gate applied right before a call cancels with the same
/// gate at the start of the callee.
TEST_F(QCOFunctionBoundaryCommutationTest,
       cancelSelfInverseGateAcrossCallBoundary) {
  programBuilder.initialize();
  programBuilder.createFunction("f", {getQubitType()}, [&](ValueRange args) {
    return SmallVector<Value>{programBuilder.h(programBuilder.x(args[0]))};
  });

  auto q = programBuilder.x(programBuilder.allocQubit());
  auto results = callFunction(programBuilder, "f", {q});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();

  // Both the caller-side and the callee-side gate disappear.
  referenceBuilder.createFunction("f_spec_boundary_commutation_arg_0",
                                  {getQubitType()}, [&](ValueRange specArgs) {
                                    return SmallVector<Value>{
                                        referenceBuilder.h(specArgs[0]),
                                    };
                                  });

  auto refQ = referenceBuilder.allocQubit();
  auto refResults = callFunction(referenceBuilder,
                                 "f_spec_boundary_commutation_arg_0", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumFunctionBoundaryCommutation());
}

/// Two different gates across the call boundary do not cancel.
TEST_F(QCOFunctionBoundaryCommutationTest, noCancellationForDifferentGates) {
  programBuilder.initialize();
  programBuilder.createFunction("f", {getQubitType()}, [&](ValueRange args) {
    return SmallVector<Value>{programBuilder.y(args[0])};
  });

  auto q = programBuilder.x(programBuilder.allocQubit());
  auto results = callFunction(programBuilder, "f", {q});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction(
      "f", {getQubitType()}, [&](ValueRange refArgs) {
        return SmallVector<Value>{referenceBuilder.y(refArgs[0])};
      });

  auto refQ = referenceBuilder.x(referenceBuilder.allocQubit());
  auto refResults = callFunction(referenceBuilder, "f", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumFunctionBoundaryCommutation());
}

/// Controlled gates are out of scope for boundary commutation, even when
/// the same one appears on both sides of the call. Cancelling them would
/// require reasoning about the control qubits as well.
TEST_F(QCOFunctionBoundaryCommutationTest, noCancellationForControlledGates) {
  const auto qubitType = getQubitType();

  programBuilder.initialize();
  programBuilder.createFunction(
      "f", {qubitType, qubitType}, [&](ValueRange args) {
        auto [innerControl, innerTarget] = programBuilder.cx(args[0], args[1]);
        return SmallVector<Value>{innerControl, innerTarget};
      });

  auto [q0, q1] =
      programBuilder.cx(programBuilder.y(programBuilder.allocQubit()),
                        programBuilder.y(programBuilder.allocQubit()));
  auto results = callFunction(programBuilder, "f", {q0, q1});
  programBuilder.sink(results[0]);
  programBuilder.sink(results[1]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction(
      "f", {qubitType, qubitType}, [&](ValueRange refArgs) {
        auto [refInnerControl, refInnerTarget] =
            referenceBuilder.cx(refArgs[0], refArgs[1]);
        return SmallVector<Value>{refInnerControl, refInnerTarget};
      });

  auto [refQ0, refQ1] =
      referenceBuilder.cx(referenceBuilder.y(referenceBuilder.allocQubit()),
                          referenceBuilder.y(referenceBuilder.allocQubit()));
  auto refResults = callFunction(referenceBuilder, "f", {refQ0, refQ1});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumFunctionBoundaryCommutation());
}

/// Two call sites that cancel the same gate share a single commuted copy
/// of the callee.
TEST_F(QCOFunctionBoundaryCommutationTest,
       reuseBoundaryCommutationAcrossCallSites) {
  const auto qubitType = getQubitType();

  programBuilder.initialize();
  programBuilder.createFunction("f", {qubitType}, [&](ValueRange args) {
    return SmallVector<Value>{programBuilder.h(programBuilder.x(args[0]))};
  });

  auto q0 = programBuilder.x(programBuilder.allocQubit());
  auto q1 = programBuilder.x(programBuilder.allocQubit());
  auto results0 = callFunction(programBuilder, "f", {q0});
  auto results1 = callFunction(programBuilder, "f", {q1});
  programBuilder.sink(results0[0]);
  programBuilder.sink(results1[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction("f_spec_boundary_commutation_arg_0",
                                  {qubitType}, [&](ValueRange specArgs) {
                                    return SmallVector<Value>{
                                        referenceBuilder.h(specArgs[0]),
                                    };
                                  });

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refResults0 = callFunction(referenceBuilder,
                                  "f_spec_boundary_commutation_arg_0", {refQ0});
  auto refResults1 = callFunction(referenceBuilder,
                                  "f_spec_boundary_commutation_arg_0", {refQ1});
  referenceBuilder.sink(refResults0[0]);
  referenceBuilder.sink(refResults1[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumFunctionBoundaryCommutation());
}

/// Two call sites that cancel a gate on different parameters of the same
/// callee must get their own specialization, because the gate is removed from a
/// specific argument.
TEST_F(QCOFunctionBoundaryCommutationTest,
       separateCommutationSpecializationPerParameter) {
  const auto qubitType = getQubitType();

  const auto buildCallee = [&qubitType](QCOProgramBuilder& b, StringRef name) {
    b.createFunction(name, {qubitType, qubitType}, [&](ValueRange args) {
      return SmallVector<Value>{b.x(args[0]), b.x(args[1])};
    });
  };

  programBuilder.initialize();
  buildCallee(programBuilder, "f");

  // The first call cancels the gate on parameter 0, ...
  auto a0 = programBuilder.x(programBuilder.allocQubit());
  auto a1 = programBuilder.allocQubit();
  auto results0 = callFunction(programBuilder, "f", {a0, a1});
  // ... the second one on parameter 1.
  auto b0 = programBuilder.allocQubit();
  auto b1 = programBuilder.x(programBuilder.allocQubit());
  auto results1 = callFunction(programBuilder, "f", {b0, b1});
  programBuilder.sink(results0[0]);
  programBuilder.sink(results0[1]);
  programBuilder.sink(results1[0]);
  programBuilder.sink(results1[1]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  // Both call sites are redirected, so the original is left without callers.
  // One specialization without the gate on parameter 0, ...
  referenceBuilder.createFunction("f_spec_boundary_commutation_arg_0",
                                  {qubitType, qubitType},
                                  [&](ValueRange spec0Args) {
                                    return SmallVector<Value>{
                                        spec0Args[0],
                                        referenceBuilder.x(spec0Args[1]),
                                    };
                                  });
  // ... and one without the gate on parameter 1.
  referenceBuilder.createFunction("f_spec_boundary_commutation_arg_1",
                                  {qubitType, qubitType},
                                  [&](ValueRange spec1Args) {
                                    return SmallVector<Value>{
                                        referenceBuilder.x(spec1Args[0]),
                                        spec1Args[1],
                                    };
                                  });

  auto refA0 = referenceBuilder.allocQubit();
  auto refA1 = referenceBuilder.allocQubit();
  auto refResults0 = callFunction(
      referenceBuilder, "f_spec_boundary_commutation_arg_0", {refA0, refA1});
  auto refB0 = referenceBuilder.allocQubit();
  auto refB1 = referenceBuilder.allocQubit();
  auto refResults1 = callFunction(
      referenceBuilder, "f_spec_boundary_commutation_arg_1", {refB0, refB1});
  referenceBuilder.sink(refResults0[0]);
  referenceBuilder.sink(refResults0[1]);
  referenceBuilder.sink(refResults1[0]);
  referenceBuilder.sink(refResults1[1]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createQuantumFunctionBoundaryCommutation());
}

// ==========================================================================

} // namespace
