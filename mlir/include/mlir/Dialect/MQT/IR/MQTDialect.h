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

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LogicalResult.h>

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQT/IR/MQTDialect.h.inc" // IWYU pragma: export

namespace mlir::mqt {
/// Return whether an operation has the program entry-point marker.
[[nodiscard]] inline bool isEntryPoint(Operation* operation) {
  return operation != nullptr &&
         operation->hasAttr(MQTDialect::EntryPointAttrHelper::getNameStr());
}

/// Mark an operation as the program entry point.
void setEntryPoint(Operation* operation);

/// Remove the program entry-point marker from an operation.
void removeEntryPoint(Operation* operation);

/// Return the source-level func.func entry point, or null if there is none.
[[nodiscard]] inline func::FuncOp getEntryPoint(ModuleOp moduleOp) {
  for (auto function : moduleOp.getOps<func::FuncOp>()) {
    if (isEntryPoint(function)) {
      return function;
    }
  }
  return nullptr;
}

/// Verify metadata invariants that involve more than one operation.
[[nodiscard]] LogicalResult verifyProgramMetadata(ModuleOp moduleOp);
} // namespace mlir::mqt
