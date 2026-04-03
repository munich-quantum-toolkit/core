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

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>

using namespace mlir;
using namespace mlir::qtensor;

enum class RangeRelation : std::uint8_t { Disjoint, Overlap, Equal, Unknown };

/**
 * @brief Checks whether two index values are equivalent for matching.
 */
static bool areEquivalentIndices(Value lhs, Value rhs) {
  return getAsOpFoldResult(lhs) == getAsOpFoldResult(rhs);
}

/**
 * @brief Checks whether two slice ranges are equivalent for matching.
 */
static bool areEquivalentRanges(Value lhsOffset, Value lhsSize, Value rhsOffset,
                                Value rhsSize) {
  return areEquivalentIndices(lhsOffset, rhsOffset) &&
         areEquivalentIndices(lhsSize, rhsSize);
}

/**
 * @brief Classify the relation between a scalar index and a slice range.
 */
static RangeRelation classifyIndexAndRange(Value index, Value offset,
                                           Value size) {
  if (areEquivalentIndices(index, offset)) {
    return RangeRelation::Overlap;
  }

  const auto indexValue = getConstantIntValue(index);
  const auto offsetValue = getConstantIntValue(offset);
  const auto sizeValue = getConstantIntValue(size);
  if (!indexValue || !offsetValue || !sizeValue) {
    return RangeRelation::Unknown;
  }

  if (*indexValue < *offsetValue || *indexValue >= *offsetValue + *sizeValue) {
    return RangeRelation::Disjoint;
  }
  return RangeRelation::Overlap;
}

/**
 * @brief Classify the relation between two slice ranges.
 */
static RangeRelation classifyRanges(Value lhsOffset, Value lhsSize,
                                    Value rhsOffset, Value rhsSize) {
  if (areEquivalentRanges(lhsOffset, lhsSize, rhsOffset, rhsSize)) {
    return RangeRelation::Equal;
  }

  const auto lhsOffsetValue = getConstantIntValue(lhsOffset);
  const auto lhsSizeValue = getConstantIntValue(lhsSize);
  const auto rhsOffsetValue = getConstantIntValue(rhsOffset);
  const auto rhsSizeValue = getConstantIntValue(rhsSize);
  if (!lhsOffsetValue || !lhsSizeValue || !rhsOffsetValue || !rhsSizeValue) {
    if (areEquivalentIndices(lhsOffset, rhsOffset)) {
      return RangeRelation::Overlap;
    }
    return RangeRelation::Unknown;
  }

  const auto lhsEnd = *lhsOffsetValue + *lhsSizeValue;
  const auto rhsEnd = *rhsOffsetValue + *rhsSizeValue;
  if (lhsEnd <= *rhsOffsetValue || rhsEnd <= *lhsOffsetValue) {
    return RangeRelation::Disjoint;
  }
  return RangeRelation::Overlap;
}

/**
 * @brief Checks whether removing an extract_slice-insert_slice pair is
 * linearity-safe.
 */
static bool
isRemovableExtractSliceInsertSlicePair(InsertSliceOp insertSliceOp,
                                       ExtractSliceOp extractSliceOp) {
  return insertSliceOp.getSource() == extractSliceOp.getResult() &&
         areEquivalentRanges(insertSliceOp.getOffset(), insertSliceOp.getSize(),
                             extractSliceOp.getOffset(),
                             extractSliceOp.getSize());
}

/**
 * @brief Find a matching `qtensor.extract_slice` for an insert_slice range in
 * a tensor chain by traversing scalar and slice tensor operations.
 */
static ExtractSliceOp
findMatchingExtractSliceInTensorChain(Value tensor, Value offset, Value size) {
  Value current = tensor;
  while (Operation* definingOp = current.getDefiningOp()) {
    if (auto nestedInsertOp = llvm::dyn_cast<InsertOp>(definingOp)) {
      if (classifyIndexAndRange(nestedInsertOp.getIndex(), offset, size) !=
          RangeRelation::Disjoint) {
        return nullptr;
      }
      current = nestedInsertOp.getDest();
      continue;
    }
    if (auto nestedInsertSliceOp = llvm::dyn_cast<InsertSliceOp>(definingOp)) {
      if (classifyRanges(nestedInsertSliceOp.getOffset(),
                         nestedInsertSliceOp.getSize(), offset,
                         size) != RangeRelation::Disjoint) {
        return nullptr;
      }
      current = nestedInsertSliceOp.getDest();
      continue;
    }
    if (auto extractOp = llvm::dyn_cast<ExtractOp>(definingOp)) {
      if (classifyIndexAndRange(extractOp.getIndex(), offset, size) !=
          RangeRelation::Disjoint) {
        return nullptr;
      }
      current = extractOp.getTensor();
      continue;
    }
    if (auto extractSliceOp = llvm::dyn_cast<ExtractSliceOp>(definingOp)) {
      const auto relation = classifyRanges(
          extractSliceOp.getOffset(), extractSliceOp.getSize(), offset, size);
      if (relation == RangeRelation::Equal) {
        return extractSliceOp;
      }
      if (relation != RangeRelation::Disjoint) {
        return nullptr;
      }
      current = extractSliceOp.getTensor();
      continue;
    }

    break;
  }

  return nullptr;
}

LogicalResult InsertSliceOp::verify() {
  auto srcDim = getSource().getType().getDimSize(0);
  auto dstDim = getDest().getType().getDimSize(0);
  auto constOffset = getConstantIntValue(getOffset());
  auto constSize = getConstantIntValue(getSize());

  if (constOffset && *constOffset < 0) {
    return emitOpError("Offset must be non-negative");
  }

  if (constSize && *constSize <= 0) {
    return emitOpError("Size must be positive");
  }

  if (constSize && !ShapedType::isDynamic(srcDim)) {
    if (*constSize != srcDim) {
      return emitOpError("Size must match source dimension");
    }
  }

  if (constOffset && constSize && !ShapedType::isDynamic(dstDim)) {
    if (*constSize > dstDim || *constOffset > dstDim - *constSize) {
      return emitOpError("Offset + Size exceeds destination dimension");
    }
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

  if (extractSliceOp.getOutTensor() != insertSliceOp.getDest()) {
    return nullptr;
  }

  auto insertOffset = insertSliceOp.getOffset();
  auto extractOffset = extractSliceOp.getOffset();
  auto insertSize = insertSliceOp.getSize();
  auto extractSize = extractSliceOp.getSize();

  if (!areEquivalentRanges(insertOffset, insertSize, extractOffset,
                           extractSize)) {
    return nullptr;
  }

  return extractSliceOp.getTensor();
}

OpFoldResult InsertSliceOp::fold(FoldAdaptor /*adaptor*/) {
  if (auto result = foldInsertAfterExtractSlice(*this)) {
    return result;
  }

  return {};
}

namespace {

/**
 * @brief Remove matching `qtensor.insert_slice` and `qtensor.extract_slice`
 * pairs through commuting disjoint tensor-chain operations.
 */
struct RemoveExtractSliceInsertSlicePair final
    : OpRewritePattern<InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertSliceOp op,
                                PatternRewriter& rewriter) const override {
    auto extractSliceOp = findMatchingExtractSliceInTensorChain(
        op.getDest(), op.getOffset(), op.getSize());
    if (!extractSliceOp) {
      return failure();
    }

    if (!isRemovableExtractSliceInsertSlicePair(op, extractSliceOp)) {
      return failure();
    }

    rewriter.replaceOp(op, op.getDest());
    rewriter.replaceOp(extractSliceOp, {extractSliceOp.getTensor(), nullptr});
    return success();
  }
};

} // namespace

void InsertSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<RemoveExtractSliceInsertSlicePair>(context);
}
