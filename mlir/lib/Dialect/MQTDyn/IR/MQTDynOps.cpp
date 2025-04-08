/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
// IWYU pragma: end_keep

#include <mlir/Support/LogicalResult.h>

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTDyn/IR/MQTDynOpsDialect.cpp.inc"

void mqt::ir::dyn::MQTDynDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/MQTDyn/IR/MQTDynOpsTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MQTDyn/IR/MQTDynOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MQTDyn/IR/MQTDynOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTDyn/IR/MQTDynInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MQTDyn/IR/MQTDynOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//

namespace mqt::ir::dyn {

mlir::LogicalResult AllocOp::verify() {
  if (!getSize() && !getSizeAttr().has_value()) {
    return emitOpError() << "expected an operand or attribute for size";
  }
  if (getSize() && getSizeAttr().has_value()) {
    return emitOpError() << "expected either an operand or attribute for size";
  }
  return mlir::success();
}

mlir::LogicalResult ExtractOp::verify() {
  if (!getIndex() && !getIndexAttr().has_value()) {
    return emitOpError() << "expected an operand or attribute for index";
  }
  if (getIndex() && getIndexAttr().has_value()) {
    return emitOpError() << "expected either an operand or attribute for index";
  }
  return mlir::success();
}

} // namespace mqt::ir::dyn
