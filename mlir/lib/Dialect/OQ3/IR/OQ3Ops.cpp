/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/OQ3/IR/OQ3Ops.h"

#include "mlir/Dialect/OQ3/IR/OQ3Dialect.h" // IWYU pragma: associated

#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/SymbolTable.h>

using namespace mlir;
using namespace mlir::oq3;

#include "mlir/Dialect/OQ3/IR/OQ3OpsDialect.cpp.inc"
#include "mlir/Dialect/OQ3/IR/OQ3OpsEnums.cpp.inc"

void OQ3Dialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/OQ3/IR/OQ3OpsTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/OQ3/IR/OQ3Ops.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/OQ3/IR/OQ3OpsTypes.cpp.inc"

LogicalResult BitType::verify(function_ref<InFlightDiagnostic()> emitError,
                              const unsigned width) {
  if (width == 0) {
    return emitError() << "bit width must be greater than zero";
  }
  return success();
}

LogicalResult AngleType::verify(function_ref<InFlightDiagnostic()> emitError,
                                const unsigned width) {
  if (width == 0) {
    return emitError() << "angle width must be greater than zero";
  }
  return success();
}

LogicalResult GateOp::verify() {
  const auto type = getFunctionType();
  if (!isa<FunctionType>(type)) {
    return emitOpError("requires a function type");
  }
  const auto functionType = cast<FunctionType>(type);
  if (getBody().empty()) {
    return emitOpError("requires a body");
  }
  auto& entry = getBody().front();
  if (entry.getNumArguments() != functionType.getNumInputs()) {
    return emitOpError("body argument count does not match the gate signature");
  }
  for (const auto [argument, expected] :
       llvm::zip_equal(entry.getArgumentTypes(), functionType.getInputs())) {
    if (argument != expected) {
      return emitOpError("body argument types do not match the gate signature");
    }
  }
  if (functionType.getNumResults() != 0) {
    return emitOpError("gate definitions cannot return classical values");
  }
  return success();
}

LogicalResult GateDeclOp::verify() {
  const auto type = dyn_cast<FunctionType>(getFunctionType());
  if (!type) {
    return emitOpError("requires a function type");
  }
  if (type.getNumResults() != 0) {
    return emitOpError("gate declarations cannot return values");
  }
  return success();
}

LogicalResult ApplyGateOp::verify() {
  Operation* declaration =
      SymbolTable::lookupNearestSymbolFrom(getOperation(), getCalleeAttr());
  if (declaration == nullptr || !isa<GateOp, GateDeclOp>(declaration)) {
    return emitOpError("references an unknown gate symbol '")
           << getCallee() << "'";
  }
  const auto functionType =
      cast<FunctionType>(isa<GateOp>(declaration)
                             ? cast<GateOp>(declaration).getFunctionType()
                             : cast<GateDeclOp>(declaration).getFunctionType());
  const auto firstQubit =
      llvm::find_if(functionType.getInputs(),
                    [](Type type) { return isa<qc::QubitType>(type); });
  const size_t parameterCount =
      std::distance(functionType.getInputs().begin(), firstQubit);
  const size_t baseQubitCount = functionType.getNumInputs() - parameterCount;
  if (getParameters().size() != parameterCount ||
      !llvm::equal(getParameters().getTypes(),
                   functionType.getInputs().take_front(parameterCount))) {
    return emitOpError(
        "operand types do not match the referenced gate signature");
  }

  const auto kinds = getModifierKinds();
  const auto indices = getModifierOperandIndices();
  if (kinds.size() != indices.size()) {
    return emitOpError("requires one operand index per gate modifier");
  }

  llvm::SmallBitVector used(getModifierOperands().size());
  size_t controlModifierCount = 0;
  for (const auto [position, rawKind] : llvm::enumerate(kinds)) {
    if (rawKind < static_cast<int32_t>(GateModifierKind::inv) ||
        rawKind > static_cast<int32_t>(GateModifierKind::pow)) {
      return emitOpError("contains an unknown gate modifier kind");
    }
    const auto kind = static_cast<GateModifierKind>(rawKind);
    controlModifierCount +=
        kind == GateModifierKind::ctrl || kind == GateModifierKind::negctrl;
    const int32_t index = indices[position];
    const bool permitsOperand = kind == GateModifierKind::pow ||
                                kind == GateModifierKind::ctrl ||
                                kind == GateModifierKind::negctrl;
    const bool requiresOperand = kind == GateModifierKind::pow;
    if (!permitsOperand && index != -1) {
      return emitOpError("inv modifiers cannot reference an operand");
    }
    if (requiresOperand && index < 0) {
      return emitOpError("pow modifiers require an exponent operand");
    }
    if (index >= 0) {
      if (static_cast<size_t>(index) >= getModifierOperands().size()) {
        return emitOpError("modifier operand index is out of bounds");
      }
      if (used.test(index)) {
        return emitOpError("modifier operands must be referenced exactly once");
      }
      used.set(index);
      const Type operandType = getModifierOperands()[index].getType();
      if ((kind == GateModifierKind::ctrl ||
           kind == GateModifierKind::negctrl) &&
          !isa<IntegerType>(operandType)) {
        return emitOpError("control modifier operands must have an integer "
                           "type");
      }
    }
  }
  if (used.count() != getModifierOperands().size()) {
    return emitOpError("contains an unreferenced modifier operand");
  }
  const size_t minimumQubitCount = baseQubitCount + controlModifierCount;
  if (getQubits().size() < minimumQubitCount ||
      (controlModifierCount == 0 && getQubits().size() != baseQubitCount)) {
    return emitOpError("qubit operands do not match the referenced gate "
                       "signature");
  }
  return success();
}

LogicalResult PackBitsOp::verify() {
  if (getBits().size() != getResult().getType().getWidth()) {
    return emitOpError("input count must match the result bit width");
  }
  return success();
}

LogicalResult UnpackBitOp::verify() {
  if (getIndex() < 0 ||
      static_cast<uint64_t>(getIndex()) >= getValue().getType().getWidth()) {
    return emitOpError("index must be within the input bit width");
  }
  return success();
}

LogicalResult ForOp::verify() {
  if (getStart().getType() != getStop().getType() ||
      getStart().getType() != getStep().getType()) {
    return emitOpError("start, stop, and step must have identical types");
  }
  if (getBody().empty() || getBody().front().getNumArguments() != 1 ||
      getBody().front().getArgument(0).getType() != getStart().getType()) {
    return emitOpError(
        "body must have one induction argument matching the range type");
  }
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/OQ3/IR/OQ3Ops.cpp.inc"
