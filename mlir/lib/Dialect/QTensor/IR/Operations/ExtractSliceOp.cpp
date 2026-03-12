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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
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

/// Verifier for ExtractSliceOp.
LogicalResult ExtractSliceOp::verify() {
  RankedTensorType sourceType = getSource().getType();

  // Element type check
  if (!llvm::isa<qco::QubitType>(sourceType.getElementType())) {
    return emitOpError("Elements of source tensor must be of qubit type");
  }

  auto srcDim = sourceType.getDimSize(0);

  if (auto constSize = getConstantIntValue(getSize())) {
    if (*constSize < 0) {
      return emitOpError("Size must be non-negative");
    }

    // Check size fits in source
    if (!ShapedType::isDynamic(srcDim) && *constSize > srcDim) {
      return emitOpError("Size exceeds source dimension");
    }

    if (auto constOffset = getConstantIntValue(getOffset())) {
      if (*constOffset < 0) {
        return emitOpError("Offset must be non-negative");
      }

      if (!ShapedType::isDynamic(srcDim) &&
          *constOffset + *constSize > srcDim) {
        return emitOpError("Offset + Size exceeds source dimension");
      }
    }
  } else if (auto constOffset = getConstantIntValue(getOffset())) {
    if (*constOffset < 0) {
      return emitOpError("Offset must be non-negative");
    }
    if (!ShapedType::isDynamic(srcDim) && *constOffset >= srcDim) {
      return emitOpError("Offset out of bounds");
    }
  }

  // Verify result slice type matches source element type
  RankedTensorType resultType = getOutSource().getType(); // or getResult()
  if (resultType.getElementType() != sourceType.getElementType()) {
    return emitOpError("result element type must match source element type");
  }

  return success();
}

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
