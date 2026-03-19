/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/IR/QCOps.h"

#include "mlir/Dialect/QC/IR/QCDialect.h" // IWYU pragma: associated

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>
// IWYU pragma: end_keep

using namespace mlir;
using namespace mlir::qc;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QC/IR/QCOpsDialect.cpp.inc"

void QCDialect::initialize() {
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/QC/IR/QCOpsTypes.cpp.inc"

      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/QC/IR/QCOps.cpp.inc"

      >();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

/// Print `!qc.qubit` (dynamic, default) or `!qc.qubit<static>`.
void QubitType::print(AsmPrinter& printer) const {
  if (getIsStatic()) {
    printer << "<static>";
  }
}

/// Parse `!qc.qubit` or `!qc.qubit<static>`.
Type QubitType::parse(AsmParser& parser) {
  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseKeyword("static") || parser.parseGreater()) {
      return {};
    }
    return get(parser.getContext(), /*isStatic=*/true);
  }
  return get(parser.getContext(), /*isStatic=*/false);
}

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/QC/IR/QCOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QC/IR/QCInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/QC/IR/QCOps.cpp.inc"
