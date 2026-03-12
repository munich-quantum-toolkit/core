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
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <optional>

using namespace mlir;
using namespace mlir::qtensor;

// Adjusted from
// https://github.com/llvm/llvm-project/blob/llvmorg-22.1.0/mlir/lib/Dialect/Tensor/IR/TensorOps.cpp

/// Verifier for InsertSliceOp.
LogicalResult InsertSliceOp::verify() {
  auto sourceType = getSource().getType();
  auto destType = getDest().getType();

  if (!llvm::isa<qco::QubitType>(sourceType.getElementType())) {
    return emitOpError("Elements of source tensor must be of qubit type");
  }
  if (!llvm::isa<qco::QubitType>(destType.getElementType())) {
    return emitOpError("Elements of dest tensor must be of qubit type");
  }

  auto dstDim = destType.getDimSize(0);
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

      // Check slice fits in dest
      if (!ShapedType::isDynamic(dstDim) &&
          *constOffset + *constSize > dstDim) {
        return emitOpError("Offset + Size exceeds destination dimension");
      }
    }
  } else if (auto constOffset = getConstantIntValue(getOffset())) {
    if (*constOffset < 0) {
      return emitOpError("Offset must be non-negative");
    }
    if (!ShapedType::isDynamic(dstDim) && *constOffset >= dstDim) {
      return emitOpError("Offset out of bounds");
    }
  }

  return success();
}

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

static Value foldIdentity(InsertSliceOp insertSliceOp) {
  auto offsetValue = getConstantIntValue(insertSliceOp.getOffset());
  auto sizeValue = getConstantIntValue(insertSliceOp.getSize());
  auto source = insertSliceOp.getSource();
  auto dest = insertSliceOp.getDest();

  if (source != dest) {
    return nullptr;
  }
  if (!offsetValue || !sizeValue) {
    return nullptr;
  }
  if (*offsetValue != 0) {
    return nullptr;
  }
  if (*sizeValue != source.getType().getDimSize(0)) {
    return nullptr;
  }
  return source;
}

OpFoldResult InsertSliceOp::fold(FoldAdaptor /*adaptor*/) {
  if (auto result = foldIdentity(*this)) {
    return result;
  }

  if (auto result = foldInsertAfterExtractSlice(*this)) {
    return result;
  }

  if (auto constSize = getConstantIntValue(getSize())) {
    if (*constSize == 0) {
      return getDest();
    }
  }

  return {};
}
namespace {

struct InsertSliceAfterInsertSlice final
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
  results.add<InsertSliceAfterInsertSlice>(context);
}
