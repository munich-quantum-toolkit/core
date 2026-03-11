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
#include <cstdint>

using namespace mlir;
using namespace mlir::qtensor;

// Adjusted from
// https://github.com/llvm/llvm-project/blob/llvmorg-22.1.0/mlir/lib/Dialect/Tensor/IR/TensorOps.cpp

/// Build an ExtractSliceOp with dynamic entries and inferred result type.
void ExtractSliceOp::build(OpBuilder& b, OperationState& result, Value source,
                           Value offset, Value size,
                           ArrayRef<NamedAttribute> attrs) {
  auto optionalVal = getConstantIntValue(size);
  auto resultType = RankedTensorType::get(
      {optionalVal ? *optionalVal : ShapedType::kDynamic},
      cast<RankedTensorType>(source.getType()).getElementType());

  result.addAttributes(attrs);
  build(b, result, {resultType, source.getType()}, source, offset, size);
}

/// Verifier for ExtractSliceOp.
LogicalResult ExtractSliceOp::verify() {
  RankedTensorType sourceType = getSourceType();

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

/**
 * @brief Rewrite pattern that pushes tensor.cast past tensor.extract_slice.
 *
 * @details
 * This pattern rewrites a `qtensor.extract_slice` operation whose source
 * operand is produced by a `tensor.cast`. When `canFoldIntoConsumerOp`
 * evaluates to true, the cast operation is moved after the slice operation.
 *
 * Conceptually, the slice is applied to the original tensor before the
 * cast, avoiding unnecessary intermediate casts.
 *
 * Example:
 *   %0 = tensor.cast %V : tensor<3x!qco.qubit> to tensor<?x!qco.qubit>
 *   %1, %2 = qtensor.extract_slice %0[0][2][1]
 *        : tensor<?x!qco.qubit> to tensor<2x!qco.qubit>
 *
 * is rewritten into:
 *
 *   %0, %1 = qtensor.extract_slice %V[0][2][1]
 *        : tensor<3x!qco.qubit> to tensor<2x!qco.qubit>
 *   %2 = tensor.cast %0 : tensor<2x!qco.qubit> to tensor<2x!qco.qubit>
 *
 * This effectively folds the cast into the consumer operation and enables
 * further canonicalization opportunities.
 */
namespace {

class ExtractSliceOpCastFolder final : public OpRewritePattern<ExtractSliceOp> {
public:
  using OpRewritePattern<ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractSliceOp sliceOp,
                                PatternRewriter& rewriter) const override {

    // Let constant folding handle constant operands
    if (matchPattern(sliceOp.getOffset(), matchConstantIndex()) ||
        matchPattern(sliceOp.getSize(), matchConstantIndex())) {
      return failure();
    }

    // Look for tensor.cast producer
    auto castOp = sliceOp.getSource().getDefiningOp<tensor::CastOp>();
    if (!castOp) {
      return failure();
    }

    if (!canFoldIntoConsumerOp(castOp)) {
      return failure();
    }

    // Verify bounds using the original tensor
    auto srcType = cast<RankedTensorType>(castOp.getSource().getType());

    int64_t dim = srcType.getShape()[0];

    auto offsetVal = getConstantIntValue(sliceOp.getOffset());
    auto sizeVal = getConstantIntValue(sliceOp.getSize());

    if (offsetVal && sizeVal) {
      if (*offsetVal + *sizeVal > dim) {
        return failure();
      }
    }

    Location loc = sliceOp.getLoc();

    // Create new slice directly on the original tensor
    auto newSlice = rewriter.create<ExtractSliceOp>(
        loc, sliceOp.getResult().getType(), sliceOp.getOutSource().getType(),
        castOp.getSource(), sliceOp.getOffset(), sliceOp.getSize());

    Value newResult = newSlice.getResult();
    Value newOutSource = newSlice.getOutSource();

    // Preserve expected type of out_source
    if (newOutSource.getType() != sliceOp.getOutSource().getType()) {
      newOutSource = rewriter.create<tensor::CastOp>(
          loc, sliceOp.getOutSource().getType(), newOutSource);
    }

    rewriter.replaceOp(sliceOp, {newResult, newOutSource});

    if (castOp->use_empty()) {
      rewriter.eraseOp(castOp);
    }

    return success();
  }
};

} // namespace

void ExtractSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                 MLIRContext* context) {
  results.add<ExtractSliceOpCastFolder>(context);
}

static InsertSliceOp foldExtractAfterInsertSlice(ExtractSliceOp extractOp) {
  auto insertOp = extractOp.getSource().getDefiningOp<InsertSliceOp>();

  if (insertOp && insertOp.getSource().getType() == extractOp.getType()) {
    return insertOp;
  }

  return nullptr;
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
