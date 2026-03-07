/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

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
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <optional>
#include <type_traits>

using namespace mlir;
using namespace mlir::qtensor;

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
static SliceVerificationResult
verifyInsertSliceOp(RankedTensorType srcType, RankedTensorType dstType,
                    ArrayRef<int64_t> /*staticOffsets*/,
                    ArrayRef<int64_t> staticSizes,
                    ArrayRef<int64_t> /*staticStrides*/,
                    RankedTensorType* expectedType = nullptr) {
  // insert_slice is the inverse of extract_slice, use the same type
  // inference.
  RankedTensorType expected =
      ExtractSliceOp::inferResultType(dstType, staticSizes);
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
 *   %0 = qtensor.insert_slice %slice0 into %input[0][2][1]
 *   %1 = qtensor.insert_slice %slice1 into %0[0][2][1]
 * ```
 *
 * This folds into:
 *
 * ```mlir
 *   %1 = qtensor.insert_slice %slice1 into %input[0][2][1]
 * ```
 */
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
        mixedSizes);
    Value toInsert = insertSliceOp.getSource();
    if (sourceType != insertSliceOp.getSourceType()) {
      OpBuilder::InsertionGuard g(rewriter);
      // The only difference between InsertSliceOp and ParallelInsertSliceOp
      // is that the insertion point is just before the InParallelOp in
      // the parallel case.

      toInsert = tensor::CastOp::create(rewriter, insertSliceOp.getLoc(),
                                        sourceType, toInsert);
    }
    rewriter.replaceOpWithNewOp<InsertOpTy>(
        insertSliceOp, toInsert, insertSliceOp.getDest(), mixedOffsets,
        mixedSizes, mixedStrides);
    return success();
  }
};

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

    Operation* replacement =
        InsertOpTy::create(rewriter, insertSliceOp.getLoc(), src, dst,
                           insertSliceOp.getMixedOffsets(), mixedSizes,
                           insertSliceOp.getMixedStrides());

    // In the parallel case there is no result and so nothing to cast.
    bool isParallelInsert =
        std::is_same_v<InsertOpTy, tensor::ParallelInsertSliceOp>;
    if (!isParallelInsert && dst.getType() != insertSliceOp.getDestType()) {
      replacement = tensor::CastOp::create(rewriter, insertSliceOp.getLoc(),
                                           insertSliceOp.getDestType(),
                                           replacement->getResult(0));
    }
    rewriter.replaceOp(insertSliceOp, replacement->getResults());
    return success();
  }
};

/**
 * @brief Inserts a tensor.cast before insert_slice when additional static
 * type information can be inferred from the slice sizes.
 *
 * @details
 * If the size operands of an insert_slice operation provide additional
 * static shape information, an explicit tensor.cast is inserted on the
 * source operand to refine its type. This enables further canonicalization
 * patterns that match tensor.cast operations, such as
 * `ForOpTensorCastFolder` in the SCF dialect.
 *
 * Example:
 *
 * ```mlir
 *   %r = qtensor.insert_slice %0 into %1[...] [3] [1]
 *       : tensor<?x!qco.qubit> into ...
 * ```
 *
 * This folds into:
 *
 * ```mlir
 *   %tmp = tensor.cast %0 : tensor<?x!qco.qubit> to tensor<3x!qco.qubit>
 *   %r = qtensor.insert_slice %tmp into %1[...] [3] [1]
 *       : tensor<3x!qco.qubit> into ...
 * ```
 */
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
    Value cast = tensor::CastOp::create(rewriter, insertSliceOp.getLoc(),
                                        newSrcType, insertSliceOp.getSource());
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
