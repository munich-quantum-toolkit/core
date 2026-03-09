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
#include <mlir/IR/Location.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cassert>
#include <cstdint>
#include <tuple>

using namespace mlir;
using namespace mlir::qtensor;

// Adjusted from
// https://github.com/llvm/llvm-project/blob/llvmorg-22.1.0/mlir/lib/Dialect/Tensor/IR/TensorOps.cpp

/**
 * @brief Infers the result type of an extract_slice operation when it is
 * not rank-reduced.
 *
 * @details
 * The result type can be inferred from the source tensor type and the
 * static representation of offsets, sizes, and strides. Special sentinel
 * values are used to encode dynamic entries.
 *
 * @param sourceTensorType The ranked source tensor type.
 * @param staticSizes The static sizes of the slice (sentinel values
 * indicate dynamic sizes).
 * @return The inferred RankedTensorType for the resulting slice.
 */
RankedTensorType
ExtractSliceOp::inferResultType(RankedTensorType sourceTensorType,
                                ArrayRef<int64_t> staticSizes) {
  // An extract_slice op may specify only a leading subset of offset/sizes/
  // strides in which case we complete with offset=0, sizes from memref type
  // and strides=1.
  assert(static_cast<int64_t>(staticSizes.size()) ==
             sourceTensorType.getRank() &&
         "unexpected staticSizes not equal to rank of source");
  return RankedTensorType::get(staticSizes, sourceTensorType.getElementType(),
                               sourceTensorType.getEncoding());
}

RankedTensorType
ExtractSliceOp::inferResultType(RankedTensorType sourceTensorType,
                                ArrayRef<OpFoldResult> sizes) {
  SmallVector<int64_t> staticSizes;
  std::tie(staticSizes, std::ignore) = decomposeMixedValues(sizes);

  assert(static_cast<int64_t>(staticSizes.size()) ==
             sourceTensorType.getRank() &&
         "unexpected staticSizes not equal to rank of source");
  return RankedTensorType::get(staticSizes, sourceTensorType.getElementType(),
                               sourceTensorType.getEncoding());
}

/**
 * @brief Computes the rank-reduced result type.
 *
 * @details
 * If the desired result rank is smaller than the number of slice sizes,
 * rank reduction is performed by dropping dimensions of size 1 until the
 * desired rank is reached.
 *
 * Multiple rank-reduced shapes may be possible. For example, a tensor of
 * shape 1x6x1 can be reduced to either 1x6 or 6x1. To ensure deterministic
 * behavior, this function always drops the first occurrences of size-1
 * dimensions.
 */
RankedTensorType ExtractSliceOp::inferCanonicalRankReducedResultType(
    unsigned desiredResultRank, RankedTensorType sourceRankedTensorType,
    ArrayRef<int64_t> sizes) {
  // Type inferred in the absence of rank-reducing behavior.
  auto inferredType = llvm::cast<RankedTensorType>(
      inferResultType(sourceRankedTensorType, sizes));
  int64_t rankDiff = inferredType.getRank() - desiredResultRank;
  if (rankDiff > 0) {
    auto shape = inferredType.getShape();
    llvm::SmallBitVector dimsToProject =
        getPositionsOfShapeOne(rankDiff, shape);
    SmallVector<int64_t> projectedShape;
    // Best effort rank-reducing: drop 1s in order.
    for (unsigned pos = 0, e = shape.size(); pos < e; ++pos) {
      if (!dimsToProject.test(pos)) {
        projectedShape.push_back(shape[pos]);
      }
    }
    inferredType =
        RankedTensorType::get(projectedShape, inferredType.getElementType());
  }
  return inferredType;
}

RankedTensorType ExtractSliceOp::inferCanonicalRankReducedResultType(
    unsigned desiredResultRank, RankedTensorType sourceRankedTensorType,
    ArrayRef<OpFoldResult> sizes) {
  SmallVector<int64_t> staticSizes;
  SmallVector<Value> dynamicSizes;
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  return ExtractSliceOp::inferCanonicalRankReducedResultType(
      desiredResultRank, sourceRankedTensorType, staticSizes);
}

/// Build an ExtractSliceOp with mixed static and dynamic entries and custom
/// result type. If the type passed is nullptr, it is inferred.
void ExtractSliceOp::build(OpBuilder& b, OperationState& result,
                           RankedTensorType resultType,
                           RankedTensorType outSourceType, Value source,
                           ArrayRef<OpFoldResult> offsets,
                           ArrayRef<OpFoldResult> sizes,
                           ArrayRef<OpFoldResult> strides,
                           ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticOffsets;
  SmallVector<int64_t> staticSizes;
  SmallVector<int64_t> staticStrides;
  SmallVector<Value> dynamicOffsets;
  SmallVector<Value> dynamicSizes;
  SmallVector<Value> dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  auto sourceRankedTensorType = llvm::cast<RankedTensorType>(source.getType());
  // Structuring implementation this way avoids duplication between builders.
  if (!resultType) {
    resultType = llvm::cast<RankedTensorType>(
        ExtractSliceOp::inferResultType(sourceRankedTensorType, staticSizes));
  }
  result.addAttributes(attrs);
  build(b, result, {resultType, outSourceType}, source, dynamicOffsets,
        dynamicSizes, dynamicStrides, b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticSizes),
        b.getDenseI64ArrayAttr(staticStrides));
}

void ExtractSliceOp::build(OpBuilder& b, OperationState& result,
                           RankedTensorType resultType, Value source,
                           ArrayRef<OpFoldResult> offsets,
                           ArrayRef<OpFoldResult> sizes,
                           ArrayRef<OpFoldResult> strides,
                           ArrayRef<NamedAttribute> attrs) {
  build(b, result, resultType, cast<RankedTensorType>(source.getType()), source,
        offsets, sizes, strides, attrs);
}

/// Build an ExtractSliceOp with mixed static and dynamic entries and inferred
/// result type.
void ExtractSliceOp::build(OpBuilder& b, OperationState& result, Value source,
                           ArrayRef<OpFoldResult> offsets,
                           ArrayRef<OpFoldResult> sizes,
                           ArrayRef<OpFoldResult> strides,
                           ArrayRef<NamedAttribute> attrs) {
  build(b, result, RankedTensorType(), cast<RankedTensorType>(source.getType()),
        source, offsets, sizes, strides, attrs);
}

/// Build an ExtractSliceOp with mixed static and dynamic entries packed into
/// a Range vector.
void ExtractSliceOp::build(OpBuilder& b, OperationState& result, Value source,
                           ArrayRef<Range> ranges,
                           ArrayRef<NamedAttribute> attrs) {
  auto [offsets, sizes, strides] = getOffsetsSizesAndStrides(ranges);
  build(b, result, RankedTensorType(), cast<RankedTensorType>(source.getType()),
        source, offsets, sizes, strides, attrs);
}

/// Build an ExtractSliceOp with dynamic entries and custom result type. If
/// the type passed is nullptr, it is inferred.
void ExtractSliceOp::build(OpBuilder& b, OperationState& result,
                           RankedTensorType resultType,
                           RankedTensorType outSourceType, Value source,
                           ValueRange offsets, ValueRange sizes,
                           ValueRange strides, ArrayRef<NamedAttribute> attrs) {
  SmallVector<OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [](Value v) -> OpFoldResult { return v; }));
  build(b, result, resultType, outSourceType, source, offsetValues, sizeValues,
        strideValues, attrs);
}
void ExtractSliceOp::build(OpBuilder& b, OperationState& result,
                           RankedTensorType resultType, Value source,
                           ValueRange offsets, ValueRange sizes,
                           ValueRange strides, ArrayRef<NamedAttribute> attrs) {
  build(b, result, resultType, cast<RankedTensorType>(source.getType()), source,
        offsets, sizes, strides, attrs);
}

/// Build an ExtractSliceOp with dynamic entries and inferred result type.
void ExtractSliceOp::build(OpBuilder& b, OperationState& result, Value source,
                           ValueRange offsets, ValueRange sizes,
                           ValueRange strides, ArrayRef<NamedAttribute> attrs) {
  build(b, result, RankedTensorType(), cast<RankedTensorType>(source.getType()),
        source, offsets, sizes, strides, attrs);
}

/// Build an ExtractSliceOp with mixed static and dynamic sizes, inferred
/// result type, offsets set to 0 and strides set to 1.
void ExtractSliceOp::build(OpBuilder& b, OperationState& result,
                           RankedTensorType resultType, Value source,
                           ArrayRef<OpFoldResult> sizes,
                           ArrayRef<NamedAttribute> attrs) {
  Attribute zeroIdxAttr = b.getIndexAttr(0);
  Attribute oneIdxAttr = b.getIndexAttr(1);
  SmallVector<OpFoldResult> readStrides(sizes.size(), oneIdxAttr);
  SmallVector<OpFoldResult> readOffsets(sizes.size(), zeroIdxAttr);
  build(b, result, resultType, cast<RankedTensorType>(source.getType()), source,
        readOffsets, sizes, readStrides, attrs);
}

/// Verifier for ExtractSliceOp.
LogicalResult ExtractSliceOp::verify() {
  RankedTensorType sourceType = getSourceType();

  if (!llvm::isa<qco::QubitType>(sourceType.getElementType())) {
    return emitOpError("Elements of tensor must be of qubit type");
  }
  // Verify result type against inferred type.
  RankedTensorType expectedType =
      ExtractSliceOp::inferResultType(sourceType, getMixedSizes());
  SliceVerificationResult result = isRankReducedType(expectedType, getType());
  if (result != SliceVerificationResult::Success) {
    return produceSliceErrorMsg(result, *this, expectedType);
  }

  // Verify that offsets, sizes, strides do not run out-of-bounds with respect
  // to the source tensor.
  SliceBoundsVerificationResult boundsResult = verifyInBoundsSlice(
      sourceType.getShape(), getStaticOffsets(), getStaticSizes(),
      getStaticStrides(), /*generateErrorMessage=*/true);
  if (!boundsResult.isValid) {
    return getOperation()->emitError(boundsResult.errorMessage);
  }

  return success();
}

llvm::SmallBitVector ExtractSliceOp::getDroppedDims() {
  return mlir::qtensor::getDroppedDims(getType().getShape(), getMixedSizes());
}

namespace {
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
class ExtractSliceOpCastFolder final : public OpRewritePattern<ExtractSliceOp> {
public:
  using OpRewritePattern<ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractSliceOp sliceOp,
                                PatternRewriter& rewriter) const override {
    // Any constant operand, just return to let the constant folder kick in.
    if (llvm::any_of(sliceOp.getOperands(), [](Value operand) {
          return matchPattern(operand, matchConstantIndex());
        })) {
      return failure();
    }

    auto castOp = sliceOp.getSource().getDefiningOp<tensor::CastOp>();
    if (!castOp) {
      return failure();
    }

    if (!canFoldIntoConsumerOp(castOp)) {
      return failure();
    }

    // Pattern does not apply if the produced op would not verify.
    SliceBoundsVerificationResult sliceResult = verifyInBoundsSlice(
        cast<RankedTensorType>(castOp.getSource().getType()).getShape(),
        sliceOp.getStaticOffsets(), sliceOp.getStaticSizes(),
        sliceOp.getStaticStrides());
    if (!sliceResult.isValid) {
      return failure();
    }

    // Create folded extract.
    Location loc = sliceOp.getLoc();
    auto newResult = ExtractSliceOp::create(
        rewriter, loc, sliceOp.getType(), castOp.getSource(),
        sliceOp.getOffsets(), sliceOp.getSizes(), sliceOp.getStrides(),
        sliceOp.getStaticOffsets(), sliceOp.getStaticSizes(),
        sliceOp.getStaticStrides());
    Value newOutSource = newResult->getResult(1);
    if (newOutSource.getType() != sliceOp.getOutSource().getType()) {
      newOutSource = tensor::CastOp::create(
          rewriter, loc, sliceOp.getOutSource().getType(), newOutSource);
    }
    rewriter.replaceOp(sliceOp, {newResult->getResult(0), newOutSource});
    if (castOp->use_empty()) {
      rewriter.eraseOp(castOp);
    }
    return success();
  }
};

} // namespace

/// Return the canonical type of the result of an extract_slice op.
struct SliceReturnTypeCanonicalizer {
  RankedTensorType operator()(ExtractSliceOp op,
                              ArrayRef<OpFoldResult> /*mixedOffsets*/,
                              ArrayRef<OpFoldResult> mixedSizes,
                              ArrayRef<OpFoldResult> /*mixedStrides*/) {
    return ExtractSliceOp::inferCanonicalRankReducedResultType(
        op.getType().getRank(), op.getSourceType(), mixedSizes);
  }
};

/// A canonicalizer wrapper to replace ExtractSliceOps.
struct SliceCanonicalizer {
  void operator()(PatternRewriter& rewriter, ExtractSliceOp op,
                  ExtractSliceOp newOp) {
    Value replacement = newOp.getResult();
    Value outSource = newOp.getOutSource();
    if (replacement.getType() != op.getType()) {
      replacement = tensor::CastOp::create(rewriter, op.getLoc(), op.getType(),
                                           replacement);
    }
    rewriter.replaceOp(op, {replacement, outSource});
  }
};

void ExtractSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                 MLIRContext* context) {
  results.add<
      OpWithOffsetSizesAndStridesConstantArgumentFolder<
          ExtractSliceOp, SliceReturnTypeCanonicalizer, SliceCanonicalizer>,
      ExtractSliceOpCastFolder>(context);
}

static InsertSliceOp foldExtractAfterInsertSlice(ExtractSliceOp extractOp) {
  auto insertOp = extractOp.getSource().getDefiningOp<InsertSliceOp>();

  auto isSame = [](OpFoldResult a, OpFoldResult b) { return a == b; };
  if (insertOp && insertOp.getSource().getType() == extractOp.getType() &&
      insertOp.isSameAs(extractOp, isSame)) {
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
