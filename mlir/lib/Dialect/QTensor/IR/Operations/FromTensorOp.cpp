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
#include <mlir/IR/OperationSupport.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cassert>
#include <cstdint>

using namespace mlir;
using namespace mlir::qtensor;

// Adjusted from
// https://github.com/llvm/llvm-project/blob/llvmorg-22.1.0/mlir/lib/Dialect/Tensor/IR/TensorOps.cpp

void FromElementsOp::build(OpBuilder& builder, OperationState& result,
                           ValueRange elements) {
  assert(!elements.empty() && "Expected at least one element");
  Type resultType = RankedTensorType::get(
      {static_cast<int64_t>(elements.size())}, elements.front().getType());
  build(builder, result, resultType, elements);
}

LogicalResult FromElementsOp::verify() {
  if (!llvm::isa<qco::QubitType>(getResult().getType().getElementType())) {
    return emitOpError("Result tensor must have qubit element type");
  }

  for (auto type : getElements().getTypes()) {
    if (!llvm::isa<qco::QubitType>(type)) {
      return emitOpError("Elements of ValueRange must be of qubit type");
    }
  }
  return success();
}
