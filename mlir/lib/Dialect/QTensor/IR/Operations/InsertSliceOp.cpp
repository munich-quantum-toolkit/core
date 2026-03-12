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

/**
 * @brief Folds consecutive InsertSliceOp operations writing to the same slice.
 *
 * @details
 * If two consecutive InsertSliceOp operations write to the same slice,
 * the destination of the second InsertSliceOp can be updated to the
 * destination of the first one, eliminating the intermediate operation.
 *
 * Example:
 *
 * ```mlir
 *   %0 = qtensor.insert_slice %slice0 into %input[%c0][%c2]
 *   %1 = qtensor.insert_slice %slice1 into %0[%c0][%c2]
 * ```
 *
 * This folds into:
 *
 * ```mlir
 *   %1 = qtensor.insert_slice %slice1 into %input[%c0][%c2]
 * ```
 */
static LogicalResult foldInsertAfterInsertSlice(InsertSliceOp insertOp) {
  // Check if the destination of current insert is another insert
  auto prevInsertOp = insertOp.getDest().getDefiningOp<InsertSliceOp>();
  if (!prevInsertOp) {
    return failure();
  }

  // Check source types
  if (prevInsertOp.getSource().getType() != insertOp.getSource().getType()) {
    return failure();
  }

  // Check offset and size
  auto prevOffsetOpt = getConstantIntValue(prevInsertOp.getOffset());
  auto prevSizeOpt = getConstantIntValue(prevInsertOp.getSize());
  auto curOffsetOpt = getConstantIntValue(insertOp.getOffset());
  auto curSizeOpt = getConstantIntValue(insertOp.getSize());

  // Only fold if offsets and sizes are constant and identical
  if (!prevOffsetOpt || !prevSizeOpt || !curOffsetOpt || !curSizeOpt) {
    return failure();
  }
  if (*prevOffsetOpt != *curOffsetOpt || *prevSizeOpt != *curSizeOpt) {
    return failure();
  }

  return success();
}

/**
 * @brief Folds round-trip extract/insert slice operation pairs.
 *
 * @details
 * Detects patterns where a slice is extracted from a tensor and then
 * inserted back into the same tensor at the same offset and size.
 * In such cases, the pair of operations forms a no-op and can
 * be folded to the original tensor value.
 *
 * Example:
 *
 * ```mlir
 * %slicedTensor, outTensor = qtensor.extract_slice %tensor[%c0][%c2]
 * %newTensor = qtensor.insert_slice %slicedTensor into %outTensor[%c0][%c2]
 * ```
 *
 * This can be folded into `%tensor`.
 */
static Value foldInsertAfterExtractSlice(InsertSliceOp insertSliceOp) {
  auto extractSliceOp =
      insertSliceOp.getSource().getDefiningOp<ExtractSliceOp>();
  if (!extractSliceOp) {
    return nullptr;
  }

  // Ensure the insert destination is the original source tensor of extract
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
  // Identity fold: full overwrite
  if (getSource() == getDest()) {
    if (auto constOffset = getConstantIntValue(getOffset())) {
      if (*constOffset == 0) {
        if (auto constSize = getConstantIntValue(getSize())) {
          if (*constSize == getSource().getType().getDimSize(0)) {
            return getSource();
          }
        }
      }
    }
  }
  // Fold nested insert after insert
  if (succeeded(foldInsertAfterInsertSlice(*this))) {
    return getResult();
  }

  // Fold after extract_slice
  if (auto result = foldInsertAfterExtractSlice(*this)) {
    return result;
  }

  // Zero-length insert
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
  };
};

} // namespace

void InsertSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<InsertSliceAfterInsertSlice>(context);
}
