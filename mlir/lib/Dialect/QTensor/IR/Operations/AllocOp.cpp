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

  auto resultType =
      RankedTensorType::get({sizeValue ? *sizeValue : ShapedType::kDynamic},
                            qco::QubitType::get(builder.getContext()));
  build(builder, result, resultType, size);
}

LogicalResult AllocOp::verify() {
  auto resultType = cast<RankedTensorType>(getResult().getType());
  auto sizeValue = getConstantIntValue(getSize());
  auto resultSize = resultType.getShape()[0];

  if (sizeValue && *sizeValue <= 0) {
    return emitOpError("Constant size operand must be positive");
  }
  if (!resultType.isDynamicDim(0)) {
    if (!sizeValue) {
      return emitOpError("Static result type requires constant size operand");
    }
    if (resultSize != *sizeValue) {
      return emitOpError("Constant size operand (")
             << *sizeValue << ") does not match static result size ("
             << resultSize << ")";
    }
  }

  return success();
}
