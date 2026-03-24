/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cassert>

using namespace mlir;
using namespace mlir::qtensor;

void AllocOp::build(OpBuilder& builder, OperationState& result, Value size) {
  auto sizeValue = getConstantIntValue(size);
  if (sizeValue) {
    assert(*sizeValue > 0 && "qtensor.alloc size must be positive");
  }

  auto resultType = RankedTensorType::get(
      {sizeValue ? *sizeValue : ShapedType::kDynamic},
      qco::QubitType::get(builder.getContext(), /*isStatic=*/false));
  build(builder, result, resultType, size);
}

LogicalResult AllocOp::verify() {
  auto resultType = cast<RankedTensorType>(getResult().getType());
  auto sizeValue = getConstantIntValue(getSize());
  auto resultSize = resultType.getShape()[0];

  if (sizeValue && *sizeValue <= 0) {
    return emitOpError("Constant size operand must be positive");
  }
  if (sizeValue.has_value() == resultType.isDynamicDim(0)) {
    return emitOpError("Size operand and result type must both be static or "
                       "both be dynamic, but got ")
           << (sizeValue ? "static size with dynamic result"
                         : "dynamic size with static result");
  }
  if (sizeValue && resultSize != *sizeValue) {
    return emitOpError("Constant size operand (")
           << *sizeValue << ") does not match static result size ("
           << resultSize << ")";
  }

  auto elementType = resultType.getElementType();
  if (auto qubitType = dyn_cast<qco::QubitType>(elementType);
      qubitType && qubitType.getIsStatic()) {
    return emitOpError("qtensor.alloc cannot allocate static qubits; element "
                       "type must be a dynamic qubit type (!qco.qubit)");
  }

  return success();
}
