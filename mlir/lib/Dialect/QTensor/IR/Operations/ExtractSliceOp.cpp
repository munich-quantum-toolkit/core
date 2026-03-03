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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qtensor;

void ExtractSliceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "q_extracted_slice");
}

/// An extract_slice result type can be inferred, when it is not
/// rank-reduced, from the source type and the static representation of
/// offsets, sizes and strides. Special sentinels encode the dynamic case.
RankedTensorType ExtractSliceOp::inferResultType(
    RankedTensorType sourceTensorType, ArrayRef<int64_t> /*staticOffsets*/,
    ArrayRef<int64_t> staticSizes, ArrayRef<int64_t> /*staticStrides*/) {
  // An extract_slice op may specify only a leading subset of offset/sizes/
  // strides in which case we complete with offset=0, sizes from memref type
  // and strides=1.
  assert(static_cast<int64_t>(staticSizes.size()) ==
             sourceTensorType.getRank() &&
         "unexpected staticSizes not equal to rank of source");
  return RankedTensorType::get(staticSizes, sourceTensorType.getElementType(),
                               sourceTensorType.getEncoding());
}

RankedTensorType ExtractSliceOp::inferResultType(
    RankedTensorType sourceTensorType, ArrayRef<OpFoldResult> /*offsets*/,
    ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> /*strides*/) {
  SmallVector<int64_t> staticSizes;
  std::tie(staticSizes, std::ignore) = decomposeMixedValues(sizes);
  assert(static_cast<int64_t>(staticSizes.size()) ==
             sourceTensorType.getRank() &&
         "unexpected staticSizes not equal to rank of source");
  return RankedTensorType::get(staticSizes, sourceTensorType.getElementType(),
                               sourceTensorType.getEncoding());
}

/// If the rank is reduced (i.e. the desiredResultRank is smaller than the
/// number of sizes), drop as many size 1 as needed to produce an inferred
/// type with the desired rank.
///
/// Note that there may be multiple ways to compute this rank-reduced type:
///   e.g. 1x6x1 can rank-reduce to either 1x6 or 6x1 2-D tensors.
///
/// To disambiguate, this function always drops the first 1 sizes occurrences.
RankedTensorType ExtractSliceOp::inferCanonicalRankReducedResultType(
    unsigned desiredResultRank, RankedTensorType sourceRankedTensorType,
    ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes,
    ArrayRef<int64_t> strides) {
  // Type inferred in the absence of rank-reducing behavior.
  auto inferredType = llvm::cast<RankedTensorType>(
      inferResultType(sourceRankedTensorType, offsets, sizes, strides));
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
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    ArrayRef<OpFoldResult> strides) {
  SmallVector<int64_t> staticOffsets;
  SmallVector<int64_t> staticSizes;
  SmallVector<int64_t> staticStrides;
  SmallVector<Value> dynamicOffsets;
  SmallVector<Value> dynamicSizes;
  SmallVector<Value> dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  return ExtractSliceOp::inferCanonicalRankReducedResultType(
      desiredResultRank, sourceRankedTensorType, staticOffsets, staticSizes,
      staticStrides);
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
    resultType = llvm::cast<RankedTensorType>(ExtractSliceOp::inferResultType(
        sourceRankedTensorType, staticOffsets, staticSizes, staticStrides));
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
                           ValueRange strides,
                           ArrayRef<NamedAttribute> /*attrs*/) {
  SmallVector<OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [](Value v) -> OpFoldResult { return v; }));
  build(b, result, resultType, outSourceType, source, offsetValues, sizeValues,
        strideValues);
}
void ExtractSliceOp::build(OpBuilder& b, OperationState& result,
                           RankedTensorType resultType, Value source,
                           ValueRange offsets, ValueRange sizes,
                           ValueRange strides, ArrayRef<NamedAttribute> attrs) {
  build(b, result, resultType, resultType, source, offsets, sizes, strides,
        attrs);
}

/// Build an ExtractSliceOp with dynamic entries and inferred result type.
void ExtractSliceOp::build(OpBuilder& b, OperationState& result, Value source,
                           ValueRange offsets, ValueRange sizes,
                           ValueRange strides, ArrayRef<NamedAttribute> attrs) {
  build(b, result, RankedTensorType(), cast<RankedTensorType>(source.getType()),
        source, offsets, sizes, strides, attrs);
}

/// Verifier for ExtractSliceOp.
LogicalResult ExtractSliceOp::verify() {
  RankedTensorType sourceType = getSourceType();

  // Verify result type against inferred type.
  RankedTensorType expectedType = ExtractSliceOp::inferResultType(
      sourceType, getMixedOffsets(), getMixedSizes(), getMixedStrides());
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
  return ::getDroppedDims(getType().getShape(), getMixedSizes());
}

FailureOr<Value>
ExtractSliceOp::rankReduceIfNeeded(OpBuilder& b, Location loc, Value value,
                                   ArrayRef<int64_t> desiredShape) {
  auto sourceTensorType = llvm::dyn_cast<RankedTensorType>(value.getType());
  assert(sourceTensorType && "not a ranked tensor type");
  auto sourceShape = sourceTensorType.getShape();
  if (sourceShape.equals(desiredShape)) {
    return value;
  }
  auto maybeRankReductionMask =
      mlir::computeRankReductionMask(sourceShape, desiredShape);
  if (!maybeRankReductionMask) {
    return failure();
  }
  return tensor::createCanonicalRankReducingExtractSliceOp(
      b, loc, value,
      RankedTensorType::Builder(sourceTensorType).setShape(desiredShape));
}

LogicalResult ExtractSliceOp::reifyResultShapes(
    OpBuilder& /*builder*/, ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0].reserve(getType().getRank());
  SmallVector<OpFoldResult> mixedSizes = getMixedSizes();
  llvm::SmallBitVector droppedDims = getDroppedDims();
  for (const auto& size : enumerate(mixedSizes)) {
    if (droppedDims.test(size.index())) {
      continue;
    }
    reifiedReturnShapes[0].push_back(size.value());
  }
  return success();
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
 *   %1, %2 = tensor.extract_slice %0[0][2][1]
 *        : tensor<?x!qco.qubit> to tensor<2x!qco.qubit>
 *
 * is rewritten into:
 *
 *   %0, %1 = tensor.extract_slice %V[0][2][1]
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
    auto newResult = rewriter.create<ExtractSliceOp>(
        loc, sliceOp.getType(), castOp.getSource(), sliceOp.getOffsets(),
        sliceOp.getSizes(), sliceOp.getStrides(), sliceOp.getStaticOffsets(),
        sliceOp.getStaticSizes(), sliceOp.getStaticStrides());
    rewriter.replaceOp(sliceOp, newResult->getResult(0));
    rewriter.replaceOp(castOp, newResult->getResult(1));
    return success();
  }
};

/// Slice elements from `values` into `outValues`. `counts` represents the
/// numbers of elements to stride in the original values for each dimension.
/// The output values can be used to construct a DenseElementsAttr.
template <typename IterTy, typename ElemTy>
void sliceElements(IterTy values, ArrayRef<int64_t> counts,
                   ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes,
                   ArrayRef<int64_t> strides,
                   llvm::SmallVectorImpl<ElemTy>* outValues) {
  assert(offsets.size() == sizes.size());
  assert(offsets.size() == strides.size());
  if (offsets.empty()) {
    return;
  }

  int64_t offset = offsets.front();
  int64_t size = sizes.front();
  int64_t stride = strides.front();
  if (offsets.size() == 1) {
    for (int64_t i = 0; i < size; ++i, offset += stride) {
      outValues->push_back(*(values + offset));
    }

    return;
  }

  for (int64_t i = 0; i < size; ++i, offset += stride) {
    auto begin = values + offset * counts.front();
    sliceElements<IterTy, ElemTy>(begin, counts.drop_front(),
                                  offsets.drop_front(), sizes.drop_front(),
                                  strides.drop_front(), outValues);
  }
}

} // namespace

/// Return the canonical type of the result of an extract_slice op.
struct SliceReturnTypeCanonicalizer {
  RankedTensorType operator()(ExtractSliceOp op,
                              ArrayRef<OpFoldResult> mixedOffsets,
                              ArrayRef<OpFoldResult> mixedSizes,
                              ArrayRef<OpFoldResult> mixedStrides) {
    return ExtractSliceOp::inferCanonicalRankReducedResultType(
        op.getType().getRank(), op.getSourceType(), mixedOffsets, mixedSizes,
        mixedStrides);
  }
};

/// A canonicalizer wrapper to replace ExtractSliceOps.
struct SliceCanonicalizer {
  void operator()(PatternRewriter& rewriter, ExtractSliceOp op,
                  ExtractSliceOp newOp) {
    Value replacement = newOp.getResult();
    Value outSource = newOp.getOutSource();
    if (replacement.getType() != op.getType()) {
      replacement = rewriter.create<tensor::CastOp>(op.getLoc(), op.getType(),
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

/// If we have an ExtractSliceOp consuming an InsertSliceOp with the same
/// slice, we can return the InsertSliceOp's source directly.
// TODO: This only checks the immediate producer; extend to go up the
// insert/extract chain if the slices are disjoint.
static Value foldExtractAfterInsertSlice(ExtractSliceOp extractOp) {
  auto insertOp = extractOp.getSource().getDefiningOp<InsertSliceOp>();

  auto isSame = [](OpFoldResult a, OpFoldResult b) { return a == b; };
  if (insertOp && insertOp.getSource().getType() == extractOp.getType() &&
      insertOp.isSameAs(extractOp, isSame)) {
    return insertOp.getSource();
  }

  return {};
}

LogicalResult ExtractSliceOp::fold(FoldAdaptor /*adaptor*/,
                                   SmallVectorImpl<OpFoldResult>& results) {

  if (getSourceType() == getType() &&
      succeeded(foldIdentityOffsetSizeAndStrideOpInterface(*this, getType()))) {
    results.push_back(this->getSource());
    results.push_back(getSource());
    return success();
  }
  if (Value slice = foldExtractAfterInsertSlice(*this)) {
    results.push_back(slice);
    results.push_back(getSource());
    return success();
  }

  return failure();
}

Value mlir::tensor::createCanonicalRankReducingExtractSliceOp(
    OpBuilder& b, Location loc, Value tensor, RankedTensorType targetType) {
  auto rankedTensorType = llvm::cast<RankedTensorType>(tensor.getType());
  unsigned rank = rankedTensorType.getRank();
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes = getMixedSizes(b, loc, tensor);
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
  return b.createOrFold<tensor::ExtractSliceOp>(loc, targetType, tensor,
                                                offsets, sizes, strides);
}
