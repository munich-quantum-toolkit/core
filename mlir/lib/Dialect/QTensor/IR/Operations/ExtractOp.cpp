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

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qtensor;

// Adjusted from
// https://github.com/llvm/llvm-project/blob/llvmorg-22.1.0/mlir/lib/Dialect/Tensor/IR/TensorOps.cpp

LogicalResult ExtractOp::verify() {
  auto tensorType = llvm::cast<RankedTensorType>(getTensor().getType());
  if (!llvm::isa<qco::QubitType>(tensorType.getElementType())) {
    return emitOpError("Elements of tensor must be of qubit type");
  }
  auto index = getConstantIntValue(getIndex());
  auto size = getTensor().getType().getDimSize(0);
  if (index) {
    if (index < 0) {
      return emitOpError("Index must be non-negative");
    }
    if (index >= size) {
      return emitOpError("Index exceeds tensor dimension");
    }
  }
  return success();
}

/**
 * @brief If an ExtractOp consumes an InsertOp with identical indices,
 * return the scalar from the InsertOp directly.
 */
static InsertOp foldExtractAfterInsert(ExtractOp extractOp) {
  auto insertOp = extractOp.getTensor().getDefiningOp<InsertOp>();
  if (!insertOp) {
    return nullptr;
  }

  if (insertOp.getScalar().getType() != extractOp.getType(0)) {
    return nullptr;
  }

  auto insertIndex = insertOp.getIndex();
  auto extractIndex = extractOp.getIndex();

  if (isSameIndex(insertIndex, extractIndex)) {
    return insertOp;
  }
  return nullptr;
}

LogicalResult ExtractOp::fold(FoldAdaptor /*adaptor*/,
                              SmallVectorImpl<OpFoldResult>& results) {
  if (auto insertOp = foldExtractAfterInsert(*this)) {
    results.push_back(insertOp.getScalar());
    results.push_back(insertOp.getDest());
  }

  return failure();
}
