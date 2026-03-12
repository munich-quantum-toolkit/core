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
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Attributes.h>
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

void ExtractSliceOp::build(OpBuilder& b, OperationState& result, Value source,
                           Value offset, Value size,
                           ArrayRef<NamedAttribute> attrs) {
  auto sizeValue = getConstantIntValue(size);
  auto resultType = RankedTensorType::get(
      {sizeValue ? *sizeValue : ShapedType::kDynamic},
      cast<RankedTensorType>(source.getType()).getElementType());

  result.addAttributes(attrs);
  build(b, result, {resultType, source.getType()}, source, offset, size);
}

LogicalResult ExtractSliceOp::verify() {
  auto sourceType = getSource().getType();
  auto resultType = getResult().getType();
  auto outSourceType = getOutSource().getType();
  auto srcDim = sourceType.getDimSize(0);
  auto constOffset = getConstantIntValue(getOffset());
  auto constSize = getConstantIntValue(getSize());

  if (!llvm::isa<qco::QubitType>(sourceType.getElementType())) {
    return emitOpError("Elements of source tensor must be of qubit type");
  }

  if (constOffset && *constOffset < 0) {
    return emitOpError("Offset must be non-negative");
  }

  if (constSize && *constSize < 0) {
    return emitOpError("Size must be non-negative");
  }

  if (constOffset && constSize && !ShapedType::isDynamic(srcDim)) {
    if (*constOffset + *constSize > srcDim) {
      return emitOpError("Offset + Size exceeds source dimension");
    }
  }

  if (resultType.getElementType() != sourceType.getElementType()) {
    return emitOpError("Result element type must match source element type");
  }

  if (outSourceType.getElementType() != sourceType.getElementType()) {
    return emitOpError(
        "OutSource tensor element type must match source element type");
  }

  if (constSize && !ShapedType::isDynamic(resultType.getDimSize(0))) {
    if (resultType.getDimSize(0) != *constSize) {
      return emitOpError("Result tensor dimension must match size operand");
    }
  }

  return success();
}

/**
 * @brief If an ExtractSliceOp consumes an InsertSliceOp with the same offset
 * and size, return the sourceTensor and the destTensor from the InsertSliceOp
 * directly.
 */
static InsertSliceOp
foldExtractAfterInsertSlice(ExtractSliceOp extractSliceOp) {
  auto insertSliceOp =
      extractSliceOp.getSource().getDefiningOp<InsertSliceOp>();
  if (!insertSliceOp) {
    return nullptr;
  }

  // Source types must match
  if (insertSliceOp.getSource().getType() != extractSliceOp.getType(0)) {
    return nullptr;
  }

  auto insertOffset = insertSliceOp.getOffset();
  auto extractOffset = extractSliceOp.getOffset();
  auto insertSize = insertSliceOp.getSize();
  auto extractSize = extractSliceOp.getSize();

  if (!isSameIndex(insertOffset, extractOffset) ||
      !isSameIndex(insertSize, extractSize)) {
    return nullptr;
  }

  return insertSliceOp;
}

LogicalResult ExtractSliceOp::fold(FoldAdaptor /*adaptor*/,
                                   SmallVectorImpl<OpFoldResult>& results) {
  if (auto insertOp = foldExtractAfterInsertSlice(*this)) {
    results.push_back(insertOp.getSource());
    results.push_back(insertOp.getDest());
    return success();
  }

  return failure();
}
