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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mqt/Dialect/MQT/IR/MQTDialect.h.inc" // IWYU pragma: export

namespace mlir::mqt {
/// Return whether an operation is the program entry point.
[[nodiscard]] inline bool isEntryPoint(Operation* operation) {
  return operation != nullptr &&
         operation->hasAttr(MQTDialect::EntryPointAttrHelper::getNameStr());
}

/// Mark an operation as the program entry point.
void setEntryPoint(Operation* operation);

/// Remove the program entry-point marker from an operation.
void removeEntryPoint(Operation* operation);

/// Return whether an operation defines a unitary function.
[[nodiscard]] inline bool isUnitaryFunction(Operation* operation) {
  return operation != nullptr &&
         operation->hasAttr(MQTDialect::UnitaryAttrHelper::getNameStr());
}

/// Mark a function as unitary.
void setUnitaryFunction(Operation* operation);

/// Return the program entry point, or null if the module has none.
[[nodiscard]] inline func::FuncOp getEntryPoint(ModuleOp moduleOp) {
  for (auto function : moduleOp.getOps<func::FuncOp>()) {
    if (isEntryPoint(function)) {
      return function;
    }
  }
  return nullptr;
}

/// Check dynamic quantum allocation placement.
///
/// An allocation in the entry block of the program entry point may live until
/// the program ends. Any other allocation must be released in the block that
/// allocates it. Modules without an entry point must not contain dynamic
/// quantum allocations. Nested modules have separate program scopes.
[[nodiscard]] LogicalResult verifyQuantumAllocations(ModuleOp moduleOp);

/// Check that every dynamic quantum allocation is in the entry block of the
/// program entry point.
///
/// This is stricter than `verifyQuantumAllocations` and serves consumers that
/// need all qubits allocated up front, such as mapping and QIR conversion.
/// Modules without an entry point must not contain dynamic quantum allocations.
[[nodiscard]] LogicalResult
verifyEntryBlockQuantumAllocations(ModuleOp moduleOp);

/// Check that every function returns one trailing value for each QCO qubit or
/// register argument, continuing those arguments in argument order.
[[nodiscard]] LogicalResult verifyQuantumArgumentReturns(ModuleOp moduleOp);
} // namespace mlir::mqt
