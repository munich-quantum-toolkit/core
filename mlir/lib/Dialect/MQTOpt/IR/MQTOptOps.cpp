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

namespace mqt::ir::opt {
mlir::ParseResult
parseOptOutputTypes(mlir::OpAsmParser& parser,
                    llvm::SmallVectorImpl<::mlir::Type>& out_qubits,
                    llvm::SmallVectorImpl<::mlir::Type>& pos_ctrl_out_qubits,
                    llvm::SmallVectorImpl<::mlir::Type>& neg_ctrl_out_qubits) {
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
    out_qubits.push_back(target);
    while (parser.parseOptionalComma().succeeded()) {
      if (parser.parseType(target).failed()) {
        return mlir::failure();
      }
      out_qubits.push_back(target);
    }
  }

  // Parse the control and negated control qubits if the corresponding keyword
  // exists.
  if (parser.parseOptionalKeyword("ctrl").succeeded()) {
    if (parser.parseTypeList(pos_ctrl_out_qubits).failed()) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected at least one type after `ctrl` keyword");
      return mlir::failure();
    }
  }

  if (parser.parseOptionalKeyword("nctrl").succeeded()) {
    if (parser.parseTypeList(neg_ctrl_out_qubits).failed()) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected at least one type after `ctrl` keyword");
      return mlir::failure();
    }
  }

  // If no types were parsed, this corresponds to e.g. like `mqtopt.i() %q :`
  if (out_qubits.empty() && pos_ctrl_out_qubits.empty() &&
      neg_ctrl_out_qubits.empty()) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected at least one type after `:`");
    return mlir::failure();
  }

  return mlir::success();
}

void printOptOutputTypes(mlir::OpAsmPrinter& printer, mlir::Operation* op,
                         mlir::TypeRange out_qubits,
                         mlir::TypeRange pos_ctrl_out_qubits,
                         mlir::TypeRange neg_ctrl_out_qubits) {
  if (out_qubits.empty() && pos_ctrl_out_qubits.empty() &&
      neg_ctrl_out_qubits.empty()) {
    return;
  }

  printer << ": ";

  if (!out_qubits.empty()) {
    printer.printType(out_qubits.front());
    for (auto type : llvm::drop_begin(out_qubits)) {
      printer << ", ";
      printer.printType(type);
    }
  }

  if (!pos_ctrl_out_qubits.empty()) {
    if (!out_qubits.empty()) {
      printer << " ";
    }
    printer << "ctrl ";
    printer.printType(pos_ctrl_out_qubits.front());
    for (auto type : llvm::drop_begin(pos_ctrl_out_qubits)) {
      printer << ", ";
      printer.printType(type);
    }
  }

  if (!neg_ctrl_out_qubits.empty()) {
    if (!pos_ctrl_out_qubits.empty() || !out_qubits.empty()) {
      printer << " ";
    }
    printer << "nctrl ";
    printer.printType(neg_ctrl_out_qubits.front());
    for (auto type : llvm::drop_begin(neg_ctrl_out_qubits)) {
      printer << ", ";
      printer.printType(type);
    }
  }
}
} // namespace mqt::ir::opt
