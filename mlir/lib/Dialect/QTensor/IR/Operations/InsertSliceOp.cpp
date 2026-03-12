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

  // Fold: bypass previous insert
  insertOp.getDestMutable().assign(prevInsertOp.getDest());
  return success();
}

/**
 * @brief Folds round-trip extract/insert slice operation pairs.
 *
 * @details
 * Detects patterns where a slice is extracted from a tensor and then
 * inserted back into the same tensor at the same offsets, sizes, and
 * strides. In such cases, the pair of operations forms a no-op and can
 * be folded to the original tensor value.
 *
 * Example:
 *
 * ```mlir
 * %0 = qtensor.extract_slice %val[0][2][1]
 * %1 = qtensor.insert_slice %0 into %val[0][2][1]
 * ```
 *
 * This can be folded into `%val`.
 */
static Value foldInsertAfterExtractSlice(InsertSliceOp insertOp) {
  auto extractOp = insertOp.getSource().getDefiningOp<ExtractSliceOp>();
  if (!extractOp) {
    return nullptr;
  }

  // Ensure the insert destination is the original source tensor of extract
  if (extractOp.getOutSource() != insertOp.getDest()) {
    return nullptr;
  }

  // Optionally check that the offset and size match exactly
  if (extractOp.getOffset() != insertOp.getOffset() ||
      extractOp.getSize() != insertOp.getSize()) {
    return nullptr;
  }

  return extractOp.getSource();
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

/**
 * @brief Folds tensor.cast operations with insert_slice.
 *
 * @details
 * If the source or destination tensor of an insert_slice operation is
 * produced by a tensor.cast that removes static type information, the
 * cast can be folded into the insert_slice operation.
 *
 * Example:
 *
 * ```mlir
 *   %1 = tensor.cast %0 : tensor<3!qco.qubit> to tensor<?x!qco.qubit>
 *   %2 = qtensor.insert_slice %1 into ... : tensor<?x!qco.qubit> into ...
 * ```
 *
 * This folds into:
 *
 * ```mlir
 *   %2 = qtensor.insert_slice %0 into ... : tensor<3!qco.qubit> into ...
 * ```
 *
 * When folding a cast on the destination tensor, the result of the
 * insert_slice operation is cast to preserve the original result type.
 */
struct InsertSliceOpCastFolder final : public OpRewritePattern<InsertSliceOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertSliceOp op,
                                PatternRewriter& rewriter) const override {

    auto srcCast = op.getSource().getDefiningOp<tensor::CastOp>();
    auto dstCast = op.getDest().getDefiningOp<tensor::CastOp>();

    if (!srcCast && !dstCast) {
      return failure();
    }

    Value newSrc =
        srcCast ? srcCast.getSource() : static_cast<Value>(op.getSource());
    Value newDst =
        dstCast ? dstCast.getSource() : static_cast<Value>(op.getDest());

    auto newOp =
        rewriter.create<InsertSliceOp>(op.getLoc(), op.getType(), newSrc,
                                       newDst, op.getOffset(), op.getSize());

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

void InsertSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<InsertSliceOpCastFolder>(context);
}
