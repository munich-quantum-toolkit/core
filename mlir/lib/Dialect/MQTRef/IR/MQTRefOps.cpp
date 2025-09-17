/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTRef/IR/MQTRefDialect.h"

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <algorithm>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
// IWYU pragma: end_keep

using namespace mqt::ir::common;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTRef/IR/MQTRefOpsDialect.cpp.inc"

void mqt::ir::ref::MQTRefDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/MQTRef/IR/MQTRefOpsTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MQTRef/IR/MQTRefOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MQTRef/IR/MQTRefOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTRef/IR/MQTRefInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MQTRef/IR/MQTRefOps.cpp.inc"

namespace mqt::ir::ref {
mlir::ParseResult parseRefParams(
    mlir::OpAsmParser& parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>& params,
    mlir::Attribute& staticParams, mlir::Attribute& paramsMask) {
  mlir::OpAsmParser::UnresolvedOperand operand;
  if (parser.parseOptionalOperand(operand).has_value()) {
    params.push_back(operand);
    while (parser.parseOptionalComma().succeeded()) {
      if (parser.parseOperand(operand).failed()) {
        return mlir::failure();
      }
      params.push_back(operand);
    }
  }

  if (parser.parseOptionalKeyword("static").succeeded()) {
    staticParams = mlir::DenseF64ArrayAttr::parse(parser, mlir::Type{});
  }

  if (parser.parseOptionalKeyword("mask").succeeded()) {
    paramsMask = mlir::DenseBoolArrayAttr::parse(parser, mlir::Type{});
  }

  return mlir::success();
}

void printRefParams(mlir::OpAsmPrinter& printer, mlir::Operation* op,
                    mlir::ValueRange params,
                    mlir::DenseF64ArrayAttr staticParams,
                    mlir::DenseBoolArrayAttr paramsMask) {
  bool needSpace = false;
  if (!params.empty()) {
    printer << params;
    needSpace = true;
  }

  if (staticParams) {
    if (needSpace) {
      printer << " ";
    }
    std::string staticStr;
    llvm::raw_string_ostream ostream(staticStr);
    staticParams.print(ostream);
    printer << "static " << ostream.str();
    needSpace = true;
  }

  if (paramsMask) {
    if (needSpace) {
      printer << " ";
    }
    std::string maskStr;
    llvm::raw_string_ostream ostream(maskStr);
    paramsMask.print(ostream);
    printer << "mask " << ostream.str();
  }
}
} // namespace mqt::ir::ref
