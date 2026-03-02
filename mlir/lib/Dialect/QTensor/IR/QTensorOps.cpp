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

#include "mlir/Dialect/QTensor/IR/QTensorDialect.h" // IWYU pragma: associated

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h> // for affine::AffineDialect
#include <mlir/Dialect/Arith/IR/Arith.h>      // for arith::ArithDialect
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Complex/IR/Complex.h> // for complex::ComplexDialect
#include <mlir/Dialect/Linalg/IR/RelayoutOpInterface.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Support/LLVM.h>
#include <optional>

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>
// IWYU pragma: end_keep

using namespace mlir;
using namespace mlir::qtensor;

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation* QTensorDialect::materializeConstant(OpBuilder& builder,
                                               Attribute value, Type type,
                                               Location loc) {
  if (auto op = arith::ConstantOp::materialize(builder, value, type, loc)) {
    return op;
  }
  if (complex::ConstantOp::isBuildableWith(value, type)) {
    return builder.create<complex::ConstantOp>(loc, type,
                                               llvm::cast<ArrayAttr>(value));
  }
  return nullptr;
}

/// Compute the dropped dimensions of a rank-reducing tensor.extract_slice op or
/// rank-extending tensor.insert_slice op.
static llvm::SmallBitVector getDroppedDims(ArrayRef<int64_t> reducedShape,
                                           ArrayRef<OpFoldResult> mixedSizes) {
  llvm::SmallBitVector droppedDims(mixedSizes.size());
  int64_t shapePos = reducedShape.size() - 1;

  for (const auto& size : enumerate(llvm::reverse(mixedSizes))) {
    size_t idx = mixedSizes.size() - size.index() - 1;
    // Rank-reduced dims must have a static unit dimension.
    bool isStaticUnitSize =
        isa<Attribute>(size.value()) &&
        llvm::cast<IntegerAttr>(cast<Attribute>(size.value())).getInt() == 1;

    if (shapePos < 0) {
      // There are no more dims in the reduced shape. All remaining sizes must
      // be rank-reduced dims.
      assert(isStaticUnitSize && "expected unit dim");
      droppedDims.set(idx);
      continue;
    }

    // Dim is preserved if the size is not a static 1.
    if (!isStaticUnitSize) {
      --shapePos;
      continue;
    }

    // Dim is preserved if the reduced shape dim is also 1.
    if (reducedShape[shapePos] == 1) {
      --shapePos;
      continue;
    }

    // Otherwise: Dim is dropped.
    droppedDims.set(idx);
  }

  assert(shapePos < 0 && "dimension mismatch");
  return droppedDims;
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

void ExtractOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "q_extracted");
}

LogicalResult ExtractOp::verify() {
  // Verify the # indices match if we have a ranked type.
  auto tensorType = llvm::cast<RankedTensorType>(getTensor().getType());
  if (tensorType.getRank() != static_cast<int64_t>(getIndices().size())) {
    return emitOpError("incorrect number of indices for extract_element");
  }
  return success();
}
struct ExtractFromTensorCast : public OpRewritePattern<ExtractOp> {
  using OpRewritePattern<ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp extract,
                                PatternRewriter& rewriter) const final {
    auto tensorCast = extract.getTensor().getDefiningOp<tensor::CastOp>();
    if (!tensorCast) {
      return failure();
    }
    if (!llvm::isa<RankedTensorType>(tensorCast.getSource().getType())) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<ExtractOp>(extract, tensorCast.getSource(),
                                           extract.getIndices());
    return success();
  }
};

/// If we have an ExtractOp consuming an InsertOp with the same
/// indices, we can return the InsertOp's scalar directly.
// TODO: This only checks the immediate producer; extend to go up the
// insert/extract chain if the slices are disjoint.
static Value foldExtractAfterInsert(ExtractOp extractOp) {
  auto insertOp = extractOp.getTensor().getDefiningOp<tensor::InsertOp>();

  auto isSame = [](Value a, Value b) {
    return getAsOpFoldResult(a) == getAsOpFoldResult(b);
  };
  if (insertOp && insertOp.getScalar().getType() == extractOp.getType(0) &&
      llvm::equal(insertOp.getIndices(), extractOp.getIndices(), isSame)) {
    return insertOp.getScalar();
  }
  return {};
}

LogicalResult ExtractOp::fold(FoldAdaptor adaptor,
                              SmallVectorImpl<OpFoldResult>& results) {
  // Collect the constant indices into the tensor.
  SmallVector<uint64_t, 8> indices;
  for (Attribute indice : adaptor.getIndices()) {
    if (!indice || !llvm::isa<IntegerAttr>(indice)) {
      return failure();
    }
    indices.push_back(llvm::cast<IntegerAttr>(indice).getInt());
  }

  if (Value result = foldExtractAfterInsert(*this)) {
    results.push_back(result);
    results.push_back(getTensor());
    return success();
  }

  return failure();
}

void ExtractOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<ExtractFromTensorCast>(context);
}

//===----------------------------------------------------------------------===//
// ExtractSliceOp
//===----------------------------------------------------------------------===//

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
  int rankDiff = inferredType.getRank() - desiredResultRank;
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

static LogicalResult produceSliceErrorMsg(SliceVerificationResult result,
                                          Operation* op,
                                          RankedTensorType expectedType) {
  switch (result) {
  case SliceVerificationResult::Success:
    return success();
  case SliceVerificationResult::RankTooLarge:
    return op->emitError("expected rank to be smaller or equal to ")
           << "the other rank. ";
  case SliceVerificationResult::SizeMismatch:
    return op->emitError("expected type to be ")
           << expectedType << " or a rank-reduced version. (size mismatch) ";
  case SliceVerificationResult::ElemTypeMismatch:
    return op->emitError("expected element type to be ")
           << expectedType.getElementType();
  default:
    llvm_unreachable("unexpected extract_slice op verification result");
  }
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

//
static LogicalResult
foldIdentityOffsetSizeAndStrideOpInterface(OffsetSizeAndStrideOpInterface op,
                                           ShapedType shapedType) {
  OpBuilder b(op.getContext());
  for (OpFoldResult opFold : op.getMixedOffsets()) {
    if (getConstantIntValue(opFold) != static_cast<int64_t>(0)) {
      return failure();
    }
  }
  // Rank-reducing noops only need to inspect the leading dimensions:
  // llvm::zip is appropriate.
  auto shape = shapedType.getShape();
  for (auto it : llvm::zip(op.getMixedSizes(), shape)) {
    if (getConstantIntValue(std::get<0>(it)) != std::get<1>(it)) {
      return failure();
    }
  }
  for (OpFoldResult opFold : op.getMixedStrides()) {
    if (getConstantIntValue(opFold) != static_cast<int64_t>(1)) {
      return failure();
    }
  }
  return success();
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

//===----------------------------------------------------------------------===//
// FromElementsOp
//===----------------------------------------------------------------------===//

void FromElementsOp::build(OpBuilder& builder, OperationState& result,
                           ValueRange elements) {
  assert(!elements.empty() && "expected at least one element");
  Type resultType = RankedTensorType::get(
      {static_cast<int64_t>(elements.size())}, elements.front().getType());
  build(builder, result, resultType, elements);
}

namespace {

struct ConvertFromElementsOpToTensorOp
    : public OpRewritePattern<qtensor::FromElementsOp> {
  using OpRewritePattern<qtensor::FromElementsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(qtensor::FromElementsOp fromElementsOp,
                                PatternRewriter& rewriter) const final {

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
        fromElementsOp, fromElementsOp.getElements());

    return success();
  }
};

} // namespace

void FromElementsOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                 MLIRContext* context) {
  results.add<ConvertFromElementsOpToTensorOp>(context);
}

//===----------------------------------------------------------------------===//
// InsertOp
//===----------------------------------------------------------------------===//

namespace {

struct ConvertInsertOpToTensorOp : public OpRewritePattern<qtensor::InsertOp> {
  using OpRewritePattern<qtensor::InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(qtensor::InsertOp insertOp,
                                PatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<tensor::InsertOp>(
        insertOp, insertOp.getScalar(), insertOp.getDest(),
        insertOp.getIndices());
    return success();
  }
};

struct InsertFromExtractOp : public OpRewritePattern<InsertOp> {
  using OpRewritePattern<InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp insertOp,
                                PatternRewriter& rewriter) const final {
    auto extractOp = insertOp.getScalar().getDefiningOp<qtensor::ExtractOp>();
    if (!extractOp) {
      return failure();
    }
    if (insertOp.getDest() != extractOp.getOutTensor()) {
      return failure();
    }
    if (insertOp.getIndices() != extractOp.getIndices()) {
      return failure();
    }

    rewriter.replaceOp(insertOp, extractOp.getTensor());
    rewriter.eraseOp(extractOp);
    return success();
  }
};

} // namespace

void InsertOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<InsertFromExtractOp, ConvertInsertOpToTensorOp>(context);
}

//===----------------------------------------------------------------------===//
// InsertSliceOp
//===----------------------------------------------------------------------===//
void InsertSliceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "inserted_slice");
}

// Build a InsertSliceOp with mixed static and dynamic entries.
void InsertSliceOp::build(OpBuilder& b, OperationState& result, Value source,
                          Value dest, ArrayRef<OpFoldResult> offsets,
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
  result.addAttributes(attrs);
  build(b, result, dest.getType(), source, dest, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticSizes),
        b.getDenseI64ArrayAttr(staticStrides));
}

/// Build an InsertSliceOp with mixed static and dynamic entries packed into a
/// Range vector.
void InsertSliceOp::build(OpBuilder& b, OperationState& result, Value source,
                          Value dest, ArrayRef<Range> ranges,
                          ArrayRef<NamedAttribute> attrs) {
  auto [offsets, sizes, strides] = getOffsetsSizesAndStrides(ranges);
  build(b, result, source, dest, offsets, sizes, strides, attrs);
}

// Build a InsertSliceOp with dynamic entries.
void InsertSliceOp::build(OpBuilder& b, OperationState& result, Value source,
                          Value dest, ValueRange offsets, ValueRange sizes,
                          ValueRange strides,
                          ArrayRef<NamedAttribute> /*attrs*/) {
  SmallVector<OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [](Value v) -> OpFoldResult { return v; }));
  build(b, result, source, dest, offsetValues, sizeValues, strideValues);
}

/// Rank-reducing type verification for both InsertSliceOp and
/// ParallelInsertSliceOp.
static SliceVerificationResult verifyInsertSliceOp(
    RankedTensorType srcType, RankedTensorType dstType,
    ArrayRef<int64_t> staticOffsets, ArrayRef<int64_t> staticSizes,
    ArrayRef<int64_t> staticStrides, RankedTensorType* expectedType = nullptr) {
  // insert_slice is the inverse of extract_slice, use the same type
  // inference.
  RankedTensorType expected = ExtractSliceOp::inferResultType(
      dstType, staticOffsets, staticSizes, staticStrides);
  if (expectedType != nullptr) {
    *expectedType = expected;
  }
  return isRankReducedType(expected, srcType);
}

/// Verifier for InsertSliceOp.
LogicalResult InsertSliceOp::verify() {
  // Verify result type against inferred type.
  RankedTensorType expectedType;
  SliceVerificationResult result =
      verifyInsertSliceOp(getSourceType(), getType(), getStaticOffsets(),
                          getStaticSizes(), getStaticStrides(), &expectedType);
  if (result != SliceVerificationResult::Success) {
    return produceSliceErrorMsg(result, *this, expectedType);
  }

  // Verify that offsets, sizes, strides do not run out-of-bounds with respect
  // to the destination tensor.
  SliceBoundsVerificationResult boundsResult = verifyInBoundsSlice(
      getDestType().getShape(), getStaticOffsets(), getStaticSizes(),
      getStaticStrides(), /*generateErrorMessage=*/true);
  if (!boundsResult.isValid) {
    return getOperation()->emitError(boundsResult.errorMessage);
  }

  return success();
}

/// If we have two consecutive InsertSliceOp writing to the same slice, we
/// can mutate the second InsertSliceOp's destination to the first one's.
///
/// Example:
///
/// ```mlir
///   %0 = tensor.insert_slice %slice0 into %input[0, 0] [64, 64] [1, 1]
///   %1 = tensor.insert_slice %slice1 into %0[0, 0] [64, 64] [1, 1]
/// ```
///
/// folds into:
///
/// ```mlir
///   %1 = tensor.insert_slice %slice1 into %input[0, 0] [64, 64] [1, 1]
/// ```
///
/// This pattern works with both InsertSliceOp and ParallelInsertSliceOp.
static LogicalResult foldInsertAfterInsertSlice(InsertSliceOp insertOp) {
  auto prevInsertOp = insertOp.getDest().getDefiningOp<InsertSliceOp>();

  auto isSame = [](OpFoldResult a, OpFoldResult b) { return a == b; };
  if (!prevInsertOp ||
      prevInsertOp.getSource().getType() != insertOp.getSource().getType() ||
      !prevInsertOp.isSameAs(insertOp, isSame)) {
    return failure();
  }

  insertOp.getDestMutable().assign(prevInsertOp.getDest());
  return success();
}

/// Folds round-trip extract/insert slice op pairs.
/// Example:
/// ```mlir
/// %0 = tensor.extract_slice %val[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1]
/// %1 = tensor.insert_slice %0 into %val[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1]
/// ```
/// can be folded into %val.
static Value foldInsertAfterExtractSlice(InsertSliceOp insertOp) {
  auto extractOp = insertOp.getSource().getDefiningOp<ExtractSliceOp>();
  auto isSame = [](OpFoldResult a, OpFoldResult b) { return a == b; };
  if (!extractOp || extractOp.getOutSource() != insertOp.getDest() ||
      !extractOp.isSameAs(insertOp, isSame)) {
    return nullptr;
  }
  return extractOp.getSource();
}

OpFoldResult InsertSliceOp::fold(FoldAdaptor) {
  if (getSourceType().hasStaticShape() && getType().hasStaticShape() &&
      getSourceType() == getType() &&
      succeeded(foldIdentityOffsetSizeAndStrideOpInterface(*this, getType()))) {
    return this->getSource();
  }
  if (succeeded(foldInsertAfterInsertSlice(*this))) {
    return getResult();
  }
  if (auto result = foldInsertAfterExtractSlice(*this)) {
    return result;
  }
  if (llvm::any_of(getMixedSizes(), isZeroInteger)) {
    return getDest();
  }
  return {};
}

LogicalResult InsertSliceOp::reifyResultShapes(
    OpBuilder& builder, ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
  reifiedReturnShapes.resize(1, SmallVector<OpFoldResult>(getType().getRank()));
  reifiedReturnShapes[0] = tensor::getMixedSizes(builder, getLoc(), getDest());
  return success();
}

namespace {
/// Pattern to rewrite a insert_slice op with constant arguments.
///
/// This pattern works with both InsertSliceOp and ParallelInsertSliceOp.
template <typename InsertOpTy>
class InsertSliceOpConstantArgumentFolder final
    : public OpRewritePattern<InsertOpTy> {
public:
  using OpRewritePattern<InsertOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOpTy insertSliceOp,
                                PatternRewriter& rewriter) const override {
    SmallVector<OpFoldResult> mixedOffsets(insertSliceOp.getMixedOffsets());
    SmallVector<OpFoldResult> mixedSizes(insertSliceOp.getMixedSizes());
    SmallVector<OpFoldResult> mixedStrides(insertSliceOp.getMixedStrides());

    // No constant operands were folded, just return;
    if (failed(foldDynamicOffsetSizeList(mixedOffsets)) &&
        failed(foldDynamicOffsetSizeList(mixedSizes)) &&
        failed(foldDynamicStrideList(mixedStrides))) {
      return failure();
    }

    // Pattern does not apply if the produced op would not verify.
    SliceBoundsVerificationResult sliceResult =
        verifyInBoundsSlice(insertSliceOp.getDest().getType().getShape(),
                            mixedOffsets, mixedSizes, mixedStrides);
    if (!sliceResult.isValid) {
      return failure();
    }

    // Create the new op in canonical form.
    auto sourceType = ExtractSliceOp::inferCanonicalRankReducedResultType(
        insertSliceOp.getSourceType().getRank(), insertSliceOp.getDestType(),
        mixedOffsets, mixedSizes, mixedStrides);
    Value toInsert = insertSliceOp.getSource();
    if (sourceType != insertSliceOp.getSourceType()) {
      OpBuilder::InsertionGuard g(rewriter);
      // The only difference between InsertSliceOp and ParallelInsertSliceOp
      // is that the insertion point is just before the ParallelCombiningOp in
      // the parallel case.
      if (std::is_same_v<InsertOpTy, tensor::ParallelInsertSliceOp>) {
        rewriter.setInsertionPoint(insertSliceOp->getParentOp());
      }
      toInsert = rewriter.create<tensor::CastOp>(insertSliceOp.getLoc(),
                                                 sourceType, toInsert);
    }
    rewriter.replaceOpWithNewOp<InsertOpTy>(
        insertSliceOp, toInsert, insertSliceOp.getDest(), mixedOffsets,
        mixedSizes, mixedStrides);
    return success();
  }
};

/// Fold tensor_casts with insert_slice operations. If the source or
/// destination tensor is a tensor_cast that removes static type information,
/// the cast is folded into the insert_slice operation. E.g.:
///
/// ```mlir
///   %1 = tensor.cast %0 : tensor<8x16xf32> to tensor<?x?xf32>
///   %2 = tensor.insert_slice %1 into ... : tensor<?x?xf32> into ...
/// ```
///
/// folds into:
///
/// ```mlir
///   %2 = tensor.insert_slice %0 into ... : tensor<8x16xf32> into ...
/// ```
///
/// Note: When folding a cast on the destination tensor, the result of the
/// insert_slice operation is casted to ensure that the type of the result did
/// not change.
///
/// This pattern works with both InsertSliceOp and ParallelInsertSliceOp.
template <typename InsertOpTy>
struct InsertSliceOpCastFolder final : public OpRewritePattern<InsertOpTy> {
  using OpRewritePattern<InsertOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOpTy insertSliceOp,
                                PatternRewriter& rewriter) const override {
    if (llvm::any_of(insertSliceOp.getOperands(), [](Value operand) {
          return matchPattern(operand, matchConstantIndex());
        })) {
      return failure();
    }

    auto getSourceOfCastOp = [](Value v) -> std::optional<Value> {
      auto castOp = v.getDefiningOp<tensor::CastOp>();
      if (!castOp || !canFoldIntoConsumerOp(castOp)) {
        return std::nullopt;
      }
      return castOp.getSource();
    };
    std::optional<Value> sourceCastSource =
        getSourceOfCastOp(insertSliceOp.getSource());
    std::optional<Value> destCastSource =
        getSourceOfCastOp(insertSliceOp.getDest());
    if (!sourceCastSource && !destCastSource) {
      return failure();
    }

    auto src =
        (sourceCastSource ? *sourceCastSource : insertSliceOp.getSource());
    auto dst = (destCastSource ? *destCastSource : insertSliceOp.getDest());
    auto srcType = llvm::dyn_cast<RankedTensorType>(src.getType());
    auto dstType = llvm::dyn_cast<RankedTensorType>(dst.getType());
    if (!srcType || !dstType) {
      return failure();
    }

    // The tensor.cast source could have additional static information not seen
    // in the insert slice op static sizes, so we ignore dynamic dims when
    // computing the rank reduction mask.
    SmallVector<int64_t> staticSizes(insertSliceOp.getStaticSizes());
    auto rankReductionMask = computeRankReductionMask(
        staticSizes, srcType.getShape(), /*matchDynamic=*/true);
    if (!rankReductionMask.has_value()) {
      return failure();
    }
    // Replace dimensions in the insert slice op with corresponding static dims
    // from the cast source type. If the insert slice sizes have static dims
    // that are not static in the tensor.cast source (i.e., when the cast op
    // casts a dynamic dim to static), the dim should not be replaced, and the
    // pattern will fail later in `verifyInsertSliceOp`.
    SmallVector<OpFoldResult> mixedSizes(insertSliceOp.getMixedSizes());
    int64_t rankReducedIdx = 0;
    for (auto [idx, size] : enumerate(staticSizes)) {
      if (!rankReductionMask.value().contains(idx) &&
          !srcType.isDynamicDim(rankReducedIdx)) {
        mixedSizes[idx] = getAsIndexOpFoldResult(
            rewriter.getContext(), srcType.getDimSize(rankReducedIdx));
        size = srcType.getDimSize(rankReducedIdx++);
      }
    }

    // Pattern does not apply if the produced op would not verify.
    if (verifyInsertSliceOp(srcType, dstType, insertSliceOp.getStaticOffsets(),
                            staticSizes, insertSliceOp.getStaticStrides()) !=
        SliceVerificationResult::Success) {
      return failure();
    }
    SliceBoundsVerificationResult sliceResult =
        verifyInBoundsSlice(dstType.getShape(), insertSliceOp.getMixedOffsets(),
                            mixedSizes, insertSliceOp.getMixedStrides());
    if (!sliceResult.isValid) {
      return failure();
    }

    Operation* replacement = rewriter.create<InsertOpTy>(
        insertSliceOp.getLoc(), src, dst, insertSliceOp.getMixedOffsets(),
        mixedSizes, insertSliceOp.getMixedStrides());

    // In the parallel case there is no result and so nothing to cast.
    bool isParallelInsert =
        std::is_same<InsertOpTy, tensor::ParallelInsertSliceOp>::value;
    if (!isParallelInsert && dst.getType() != insertSliceOp.getDestType()) {
      replacement = rewriter.create<tensor::CastOp>(insertSliceOp.getLoc(),
                                                    insertSliceOp.getDestType(),
                                                    replacement->getResult(0));
    }
    rewriter.replaceOp(insertSliceOp, replacement->getResults());
    return success();
  }
};

/// If additional static type information can be deduced from a insert_slice's
/// size operands, insert an explicit cast of the op's source operand. This
/// enables other canonicalization patterns that are matching for tensor_cast
/// ops such as `ForOpTensorCastFolder` in SCF.
///
/// Example:
///
/// ```mlir
///   %r = tensor.insert_slice %0 into %1[...] [64, 64] [1, 1]
///       : tensor<?x?xf32> into ...
/// ```
///
/// folds into:
///
/// ```mlir
///   %tmp = tensor.cast %0 : tensor<?x?xf32> to tensor<64x64xf32>
///   %r = tensor.insert_slice %tmp into %1[...] [64, 64] [1, 1]
///       : tensor<64x64xf32> into ...
/// ```
///
/// This patterns works with both InsertSliceOp and ParallelInsertSliceOp.
template <typename InsertOpTy>
struct InsertSliceOpSourceCastInserter final
    : public OpRewritePattern<InsertOpTy> {
  using OpRewritePattern<InsertOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOpTy insertSliceOp,
                                PatternRewriter& rewriter) const override {
    RankedTensorType srcType = insertSliceOp.getSourceType();
    if (srcType.getRank() != insertSliceOp.getDestType().getRank()) {
      return failure();
    }
    SmallVector<int64_t> newSrcShape(srcType.getShape());
    for (int64_t i = 0; i < srcType.getRank(); ++i) {
      if (std::optional<int64_t> constInt =
              getConstantIntValue(insertSliceOp.getMixedSizes()[i])) {
        // Bail on invalid IR.
        if (*constInt < 0) {
          return failure();
        }
        newSrcShape[i] = *constInt;
      }
    }
    if (!hasValidSizesOffsets(newSrcShape)) {
      return failure();
    }

    RankedTensorType newSrcType = RankedTensorType::get(
        newSrcShape, srcType.getElementType(), srcType.getEncoding());
    if (srcType == newSrcType ||
        !mlir::tensor::preservesStaticInformation(srcType, newSrcType) ||
        !tensor::CastOp::areCastCompatible(srcType, newSrcType)) {
      return failure();
    }

    // newSrcType is:
    //   1) Different from srcType.
    //   2) "More static" than srcType.
    //   3) Cast-compatible with srcType.
    // Insert the cast.
    OpBuilder::InsertionGuard g(rewriter);
    // The only difference between InsertSliceOp and ParallelInsertSliceOp is
    // that the insertion point is just before the ParallelCombiningOp in the
    // parallel case.
    if (std::is_same_v<InsertOpTy, tensor::ParallelInsertSliceOp>) {
      rewriter.setInsertionPoint(insertSliceOp->getParentOp());
    }
    Value cast = rewriter.create<tensor::CastOp>(
        insertSliceOp.getLoc(), newSrcType, insertSliceOp.getSource());
    rewriter.replaceOpWithNewOp<InsertOpTy>(
        insertSliceOp, cast, insertSliceOp.getDest(),
        insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
        insertSliceOp.getMixedStrides());
    return success();
  }
};
} // namespace

llvm::SmallBitVector InsertSliceOp::getDroppedDims() {
  return ::getDroppedDims(getSourceType().getShape(), getMixedSizes());
}

void InsertSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<InsertSliceOpConstantArgumentFolder<InsertSliceOp>,
              InsertSliceOpCastFolder<InsertSliceOp>,
              InsertSliceOpSourceCastInserter<InsertSliceOp>>(context);
}

Value mlir::tensor::createCanonicalRankReducingInsertSliceOp(OpBuilder& b,
                                                             Location loc,
                                                             Value tensor,
                                                             Value dest) {
  auto rankedTensorType = llvm::cast<RankedTensorType>(dest.getType());
  unsigned rank = rankedTensorType.getRank();
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes = getMixedSizes(b, loc, dest);
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
  return b.createOrFold<tensor::InsertSliceOp>(loc, tensor, dest, offsets,
                                               sizes, strides);
}

//===----------------------------------------------------------------------===//
// Common Canonicalizers and Folders.
//===----------------------------------------------------------------------===//

namespace {

bool foldTensorCastPrecondition(DestinationStyleOpInterface op) {
  // 1. InsertSliceOp has its own logic about folding tensor.cast ops.
  // 2. Exclude DPS ops that are also LoopLike from this interface as they
  // might need special handling of attached regions.
  if (isa<InsertSliceOp>(op.getOperation()) ||
      isa<LoopLikeOpInterface>(op.getOperation())) {
    return false;
  }

  return mlir::tensor::hasFoldableTensorCastOperand(op);
}
} // namespace

struct FoldTensorCastProducerOp
    : public OpInterfaceRewritePattern<DestinationStyleOpInterface> {
  using OpInterfaceRewritePattern<
      DestinationStyleOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(DestinationStyleOpInterface op,
                                PatternRewriter& rewriter) const override {

    // Reject PackOp/UnpackOp (i.e. RelayoutOps) - there are dedicated patterns
    // for that instead.
    if (!foldTensorCastPrecondition(op) ||
        isa<linalg::RelayoutOpInterface>(*op)) {
      return failure();
    }

    SmallVector<Type> newResultTypes(op->getResultTypes());
    SmallVector<Value> newOperands =
        mlir::tensor::getUpdatedOperandsAfterCastOpFolding(op, newResultTypes);

    // Clone op
    auto newOp = clone(rewriter, op, newResultTypes, newOperands);

    SmallVector<Value, 4> replacements;
    replacements.reserve(newOp->getNumResults());
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), newOp->getResults())) {
      if (newResult.getType() != oldResult.getType()) {
        replacements.push_back(rewriter.create<tensor::CastOp>(
            op->getLoc(), oldResult.getType(), newResult));
      } else {
        replacements.push_back(newResult);
      }
    }
    rewriter.replaceOp(op, replacements);

    return success();
  }
};

void QTensorDialect::getCanonicalizationPatterns(
    RewritePatternSet& results) const {
  results.add<FoldTensorCastProducerOp>(getContext());
}

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QTensor/IR/QTensorOpsDialect.cpp.inc"

void QTensorDialect::initialize() {
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/QTensor/IR/QTensorOpsTypes.cpp.inc"

      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/QTensor/IR/QTensorOps.cpp.inc"

      >();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/QTensor/IR/QTensorOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/QTensor/IR/QTensorOps.cpp.inc"
