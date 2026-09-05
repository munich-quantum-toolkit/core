/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <iterator>

using namespace mlir;
using namespace mlir::qc;

size_t CallOp::getNumParams() {
  return static_cast<size_t>(std::distance(
      getOperands().begin(), llvm::find_if(getOperands(), [](Value value) {
        return isa<QubitType>(value.getType());
      })));
}

size_t CallOp::getNumQubits() { return getNumOperands() - getNumParams(); }

OperandRange CallOp::getParameters() {
  return getOperands().take_front(getNumParams());
}

OperandRange CallOp::getQubits() {
  return getOperands().drop_front(getNumParams());
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
  auto function =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getCalleeAttr());
  if (!function) {
    return emitOpError() << "'" << getCallee()
                         << "' does not reference a valid function";
  }
  if (!mqt::isUnitaryFunction(function)) {
    return emitOpError() << "callee '" << getCallee()
                         << "' is not marked with mqt.unitary";
  }
  if (function.getArgumentTypes() != getOperandTypes()) {
    return emitOpError() << "operand types " << getOperandTypes()
                         << " do not match callee argument types "
                         << function.getArgumentTypes();
  }
  if (function.getNumResults() != 0) {
    return emitOpError("unitary QC callee must not return values");
  }
  return success();
}
