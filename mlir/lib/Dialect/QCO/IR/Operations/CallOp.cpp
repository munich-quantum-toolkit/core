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
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <iterator>

using namespace mlir;
using namespace mlir::qco;

void CallOp::build(OpBuilder&, OperationState& state, FlatSymbolRefAttr callee,
                   ValueRange operands) {
  state.addAttribute("callee", callee);
  state.addOperands(operands);
  for (Value operand : operands) {
    if (isa<QubitType>(operand.getType())) {
      state.addTypes(operand.getType());
    }
  }
}

size_t CallOp::getNumParams() {
  return getNumOperands() < getNumResults()
             ? 0
             : getNumOperands() - getNumResults();
}

OperandRange CallOp::getParameters() {
  return getOperands().take_front(getNumParams());
}

OperandRange CallOp::getInputQubits() {
  return getOperands().drop_front(getNumParams());
}

Value CallOp::getInputForOutput(Value output) {
  auto result = dyn_cast<OpResult>(output);
  auto inputs = getInputQubits();
  if (!result || result.getOwner() != getOperation() ||
      result.getResultNumber() >= inputs.size()) {
    return {};
  }
  return inputs[result.getResultNumber()];
}

Value CallOp::getOutputForInput(Value input) {
  const auto position = llvm::find(getInputQubits(), input);
  if (position == getInputQubits().end()) {
    return {};
  }
  return getOutputQubit(
      static_cast<size_t>(std::distance(getInputQubits().begin(), position)));
}

LogicalResult CallOp::verify() {
  if (getNumOperands() < getNumResults() ||
      llvm::any_of(
          getParameters(),
          [](Value value) { return isa<QubitType>(value.getType()); }) ||
      llvm::any_of(getInputQubits(), [](Value value) {
        return !isa<QubitType>(value.getType());
      })) {
    return emitOpError(
        "requires one trailing qubit operand for every qubit result");
  }
  return success();
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
  if (function.getResultTypes() != getResultTypes()) {
    return emitOpError() << "result types " << getResultTypes()
                         << " do not match callee result types "
                         << function.getResultTypes();
  }
  return success();
}
