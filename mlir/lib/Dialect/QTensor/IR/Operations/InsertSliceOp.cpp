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
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <optional>

using namespace mlir;
using namespace mlir::qtensor;

LogicalResult InsertSliceOp::verify() {
  auto sourceType = getSource().getType();
  auto destType = getDest().getType();
  auto srcDim = sourceType.getDimSize(0);
  auto dstDim = destType.getDimSize(0);
  auto constOffset = getConstantIntValue(getOffset());
  auto constSize = getConstantIntValue(getSize());

  if (!llvm::isa<qco::QubitType>(sourceType.getElementType())) {
    return emitOpError("Elements of source tensor must be of qubit type");
  }

  if (!llvm::isa<qco::QubitType>(destType.getElementType())) {
    return emitOpError("Elements of dest tensor must be of qubit type");
  }

  if (constOffset && *constOffset < 0) {
    return emitOpError("Offset must be non-negative");
  }

  if (constSize && *constSize < 0) {
    return emitOpError("Size must be non-negative");
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

  if (getResult().getType() != destType) {
    return emitOpError("Result type must match dest type");
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

  if (extractSliceOp.getOutSource() != insertSliceOp.getDest()) {
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

  return extractSliceOp.getSource();
}

OpFoldResult InsertSliceOp::fold(FoldAdaptor /*adaptor*/) {
  if (auto result = foldInsertAfterExtractSlice(*this)) {
    return result;
  }

  // Fold InsertSliceOp if size is 0
  if (auto constSize = getConstantIntValue(getSize())) {
    if (*constSize == 0) {
      return getDest();
    }
  }

  return {};
}

namespace {

/**
 * @brief Combine subsequent insertSlice operations with the same offset and
 * size.
 */
struct CombineSubsequentInsertSliceOp final
    : public OpRewritePattern<InsertSliceOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertSliceOp insertSliceOp,
                                PatternRewriter& rewriter) const override {
    auto prevInsertOp = insertSliceOp.getDest().getDefiningOp<InsertSliceOp>();
    if (!prevInsertOp) {
      return failure();
    }

    // Source types must match
    if (prevInsertOp.getSource().getType() !=
        insertSliceOp.getSource().getType()) {
      return failure();
    }

    auto prevOffset = prevInsertOp.getOffset();
    auto curOffset = insertSliceOp.getOffset();
    auto prevSize = prevInsertOp.getSize();
    auto curSize = insertSliceOp.getSize();

    if (!isSameIndex(prevOffset, curOffset) ||
        !isSameIndex(prevSize, curSize)) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<InsertSliceOp>(
        insertSliceOp, insertSliceOp.getSource(), prevInsertOp.getDest(),
        curOffset, curSize);
    rewriter.eraseOp(prevInsertOp);
    return success();
  }
};

} // namespace

void InsertSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<CombineSubsequentInsertSliceOp>(context);
}
