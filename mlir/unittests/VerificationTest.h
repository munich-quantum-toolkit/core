/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mqt/Dialect/QCO/Utils/Verification.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstddef>

namespace mqt::test {

/// Select an explicit main or the sole defined function in a test module.
inline mlir::func::FuncOp verificationEntry(mlir::ModuleOp moduleOp) {
  if (auto main = moduleOp.lookupSymbol<mlir::func::FuncOp>("main")) {
    return main.isDeclaration() ? mlir::func::FuncOp{} : main;
  }
  mlir::func::FuncOp entry;
  for (auto func : moduleOp.getBody()->getOps<mlir::func::FuncOp>()) {
    if (func.isDeclaration()) {
      continue;
    }
    if (entry) {
      return {};
    }
    entry = func;
  }
  return entry;
}

/// Check every matrix entry through native construction, preserving phase.
///
/// The tolerance remains the maximum absolute entry residual. These tests
/// preserve phase because the rewritten function may later acquire controls.
inline void expectUnitaryEquivalent(mlir::ModuleOp expectedModule,
                                    mlir::ModuleOp actualModule,
                                    size_t numQubits,
                                    double tolerance = 1e-12) {
  auto expected = verificationEntry(expectedModule);
  auto actual = verificationEntry(actualModule);
  ASSERT_TRUE(expected) << "expected module needs main or one defined function";
  ASSERT_TRUE(actual) << "actual module needs main or one defined function";
  mlir::qco::VerificationContract contract;
  contract.numInputs = numQubits;
  mlir::qco::VerificationOptions options;
  options.method = mlir::qco::VerificationMethod::Construction;
  options.relation = mlir::qco::VerificationRelation::Exact;
  options.phase = mlir::qco::VerificationPhase::Preserve;
  options.tolerance = tolerance;
  const auto result = mlir::qco::verify(
      expected, actual, contract, mlir::qco::DDArgumentBindings{},
      mlir::qco::DDArgumentBindings{}, options);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->verdict, mlir::qco::VerificationVerdict::Equivalent)
      << result->reason;
}

} // namespace mqt::test
