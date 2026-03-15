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

#include <llvm/Support/Casting.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Support/LogicalResult.h>

#include <cassert>
#include <cstdint>

using namespace mlir;
using namespace mlir::qtensor;

void AllocOp::build(OpBuilder& builder, OperationState& result, int64_t size) {
  assert(size > 0 && "qtensor.alloc size must be positive");

  auto resultType =
      RankedTensorType::get({size}, qco::QubitType::get(builder.getContext()));
  build(builder, result, resultType,
        IntegerAttr::get(builder.getIntegerType(64), size));
}

LogicalResult AllocOp::verify() {
  auto resultType = getResult().getType();
  auto size = static_cast<int64_t>(getSize());

  if (resultType.getShape()[0] != size) {
    return emitOpError("Tensor length must match size attribute (")
           << size << "), but got " << resultType.getShape()[0];
  }

  return success();
}
