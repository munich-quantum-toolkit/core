/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h" // IWYU pragma: associated
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
// IWYU pragma: end_keep

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsDialect.cpp.inc"

void mqt::ir::opt::MQTOptDialect::initialize() {
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MQTOpt/IR/MQTOptOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MQTOpt/IR/MQTOptOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTOpt/IR/MQTOptInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MQTOpt/IR/MQTOptOps.cpp.inc"

namespace {
/**
 * @brief Prints the given list of types as a comma-separated list
 *
 * @param printer The printer to use.
 * @param types The types to print.
 **/
void printCommaSeparated(mlir::OpAsmPrinter& printer,
                                mlir::TypeRange types) {
  if (types.empty()) {
    return;
  }

  printer.printType(types.front());
  for (auto type : llvm::drop_begin(types)) {
    printer << ", ";
    printer.printType(type);
  }
}
} // namespace

namespace mqt::ir::opt {
mlir::ParseResult
parseOptOutputTypes(mlir::OpAsmParser& parser,
                    llvm::SmallVectorImpl<::mlir::Type>& outQubits,
                    llvm::SmallVectorImpl<::mlir::Type>& posCtrlOutQubits,
                    llvm::SmallVectorImpl<::mlir::Type>& negCtrlOutQubits) {
  // No ":" delimiter -> no output types.
  if (parser.parseOptionalColon().failed()) {
    return mlir::success();
  }

  // Also allow `: ()`
  if (parser.parseOptionalLParen().succeeded()) {
    if (parser.parseRParen().failed()) {
      return mlir::failure();
    }
    return mlir::success();
  }

  // Parse the type of the target (there is no `parseOptionalTypeList` method so
  // we need to do this manually).
  mlir::Type target;
  if (parser.parseOptionalType(target).has_value()) {
    outQubits.push_back(target);
    while (parser.parseOptionalComma().succeeded()) {
      if (parser.parseType(target).failed()) {
        return mlir::failure();
      }
      outQubits.push_back(target);
    }
  }

  // Parse the control and negated control qubits if the corresponding keyword
  // exists.
  if (parser.parseOptionalKeyword("ctrl").succeeded()) {
    if (parser.parseTypeList(posCtrlOutQubits).failed()) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected at least one type after `ctrl` keyword");
      return mlir::failure();
    }
  }

  if (parser.parseOptionalKeyword("nctrl").succeeded()) {
    if (parser.parseTypeList(negCtrlOutQubits).failed()) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected at least one type after `ctrl` keyword");
      return mlir::failure();
    }
  }

  // If no types were parsed, this corresponds to e.g. like `mqtopt.i() %q :`
  if (outQubits.empty() && posCtrlOutQubits.empty() &&
      negCtrlOutQubits.empty()) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected at least one type after `:`");
    return mlir::failure();
  }

  return mlir::success();
}

void printOptOutputTypes(mlir::OpAsmPrinter& printer, mlir::Operation* /*op*/,
                         mlir::TypeRange outQubits,
                         mlir::TypeRange posCtrlOutQubits,
                         mlir::TypeRange negCtrlOutQubits) {
  if (outQubits.empty() && posCtrlOutQubits.empty() &&
      negCtrlOutQubits.empty()) {
    return;
  }

  printer << ": ";

  printCommaSeparated(printer, outQubits);

  if (!posCtrlOutQubits.empty()) {
    if (!outQubits.empty()) {
      printer << " ";
    }
    printer << "ctrl ";
    printCommaSeparated(printer, posCtrlOutQubits);
  }

  if (!negCtrlOutQubits.empty()) {
    if (!posCtrlOutQubits.empty() || !outQubits.empty()) {
      printer << " ";
    }
    printer << "nctrl ";
    printCommaSeparated(printer, negCtrlOutQubits);
  }
}
} // namespace mqt::ir::opt
