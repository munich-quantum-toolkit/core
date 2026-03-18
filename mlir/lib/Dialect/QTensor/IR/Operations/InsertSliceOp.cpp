/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qtensor;

LogicalResult InsertSliceOp::verify() {
  auto srcDim = getSource().getType().getDimSize(0);
  auto dstDim = getDest().getType().getDimSize(0);
  auto constOffset = getConstantIntValue(getOffset());
  auto constSize = getConstantIntValue(getSize());

  if (constOffset && *constOffset < 0) {
    return emitOpError("Offset must be non-negative");
  }

  if (constSize && *constSize <= 0) {
    return emitOpError("Size must be positive");
  }

  if (constSize && !ShapedType::isDynamic(srcDim)) {
    if (*constSize != srcDim) {
      return emitOpError("Size must match source dimension");
    }
  }

  if (constOffset && constSize && !ShapedType::isDynamic(dstDim)) {
    if (*constOffset + *constSize > dstDim) {
      return emitOpError("Offset + Size exceeds destination dimension");
    }
  }

  return success();
}

/**
 * @brief If an InsertSliceOp consumes an ExtractSliceOp with the same offset
 * and size, return the sourceTensor from the extractSliceOp directly.
 */
static Value foldInsertAfterExtractSlice(InsertSliceOp insertSliceOp) {
  auto extractSliceOp =
      insertSliceOp.getSource().getDefiningOp<ExtractSliceOp>();
  if (!extractSliceOp) {
    return nullptr;
  }

  if (extractSliceOp.getOutTensor() != insertSliceOp.getDest()) {
    return nullptr;
  }

  auto insertOffset = insertSliceOp.getOffset();
  auto extractOffset = extractSliceOp.getOffset();
  auto insertSize = insertSliceOp.getSize();
  auto extractSize = extractSliceOp.getSize();

  if (getAsOpFoldResult(insertOffset) != getAsOpFoldResult(extractOffset) ||
      getAsOpFoldResult(insertSize) != getAsOpFoldResult(extractSize)) {
    return nullptr;
  }

  return extractSliceOp.getTensor();
}

OpFoldResult InsertSliceOp::fold(FoldAdaptor /*adaptor*/) {
  if (auto result = foldInsertAfterExtractSlice(*this)) {
    return result;
  }

  return {};
}
