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

#include <mlir/Support/LogicalResult.h>

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

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//

namespace mqt::ir::opt {

mlir::LogicalResult BarrierOp::verify() {
  if (!getPosCtrlQubits().empty() || !getNegCtrlQubits().empty()) {
    return emitOpError() << "Barrier gate should not have control qubits";
  }
  return mlir::success();
}

mlir::LogicalResult MeasureOp::verify() {
  if (getInQubits().size() != getOutQubits().size()) {
    return emitOpError() << "number of input qubits (" << getInQubits().size()
                         << ") " << "and output qubits ("
                         << getOutQubits().size() << ") must be the same";
  }
  return mlir::success();
}

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

mlir::LogicalResult InsertOp::verify() {
  if (!getIndex() && !getIndexAttr().has_value()) {
    return emitOpError() << "expected an operand or attribute for index";
  }
  if (getIndex() && getIndexAttr().has_value()) {
    return emitOpError() << "expected either an operand or attribute for index";
  }
  return mlir::success();
}

} // namespace mqt::ir::opt
