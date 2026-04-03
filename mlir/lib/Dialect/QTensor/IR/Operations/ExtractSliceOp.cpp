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
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
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
 * @brief Tensor-transforming ops in a chain that can commute with slice
 * extracts.
 */
static bool isTensorChainOp(Operation* op) {
  return llvm::isa<InsertOp, ExtractOp, InsertSliceOp, ExtractSliceOp>(op);
}

/**
 * @brief Returns the tensor input of a tensor-transforming op.
 */
static Value getTensorChainInput(Operation* op) {
  if (auto insertOp = llvm::dyn_cast<InsertOp>(op)) {
    return insertOp.getDest();
  }
  if (auto extractOp = llvm::dyn_cast<ExtractOp>(op)) {
    return extractOp.getTensor();
  }
  if (auto insertSliceOp = llvm::dyn_cast<InsertSliceOp>(op)) {
    return insertSliceOp.getDest();
  }
  if (auto extractSliceOp = llvm::dyn_cast<ExtractSliceOp>(op)) {
    return extractSliceOp.getTensor();
  }
  return nullptr;
}

/**
 * @brief Returns the tensor output of a tensor-transforming op.
 */
static Value getTensorChainOutput(Operation* op) {
  if (auto insertOp = llvm::dyn_cast<InsertOp>(op)) {
    return insertOp.getResult();
  }
  if (auto extractOp = llvm::dyn_cast<ExtractOp>(op)) {
    return extractOp.getOutTensor();
  }
  if (auto insertSliceOp = llvm::dyn_cast<InsertSliceOp>(op)) {
    return insertSliceOp.getResult();
  }
  if (auto extractSliceOp = llvm::dyn_cast<ExtractSliceOp>(op)) {
    return extractSliceOp.getOutTensor();
  }
  return nullptr;
}

/**
 * @brief Rewire the tensor input of a tensor-transforming op.
 */
static void setTensorChainInput(Operation* op, Value tensor) {
  if (llvm::isa<InsertOp, InsertSliceOp>(op)) {
    op->setOperand(1, tensor);
    return;
  }
  if (llvm::isa<ExtractOp, ExtractSliceOp>(op)) {
    op->setOperand(0, tensor);
  }
}

void ExtractSliceOp::build(OpBuilder& b, OperationState& result, Value tensor,
                           Value offset, Value size,
                           ArrayRef<NamedAttribute> attrs) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto sizeValue = getConstantIntValue(size);
  auto resultType = RankedTensorType::get(
      {sizeValue ? *sizeValue : ShapedType::kDynamic},
      tensorType.getElementType(), tensorType.getEncoding());

  result.addAttributes(attrs);
  build(b, result, {tensor.getType(), resultType}, tensor, offset, size);
}

LogicalResult ExtractSliceOp::verify() {
  auto tensorDim = getTensor().getType().getDimSize(0);
  auto resultDim = getResult().getType().getDimSize(0);
  auto constOffset = getConstantIntValue(getOffset());
  auto constSize = getConstantIntValue(getSize());

  if (constOffset && *constOffset < 0) {
    return emitOpError("Offset must be non-negative");
  }

  if (constSize && *constSize <= 0) {
    return emitOpError("Size must be positive");
  }

  if (constOffset && constSize && !ShapedType::isDynamic(tensorDim)) {
    if (*constOffset + *constSize > tensorDim) {
      return emitOpError("Offset + Size exceeds source dimension");
    }
  }

  if (constSize && !ShapedType::isDynamic(resultDim)) {
    if (resultDim != *constSize) {
      return emitOpError("Result tensor dimension must match size operand");
    }
  }

  return success();
}

/**
 * @brief If an ExtractSliceOp consumes an InsertSliceOp with the same offset
 * and size, return the sourceTensor and the destTensor from the InsertSliceOp
 * directly.
 */
static InsertSliceOp
foldExtractAfterInsertSlice(ExtractSliceOp extractSliceOp) {
  auto insertSliceOp =
      extractSliceOp.getTensor().getDefiningOp<InsertSliceOp>();
  if (!insertSliceOp) {
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

  return insertSliceOp;
}

LogicalResult ExtractSliceOp::fold(FoldAdaptor /*adaptor*/,
                                   SmallVectorImpl<OpFoldResult>& results) {
  if (auto insertOp = foldExtractAfterInsertSlice(*this)) {
    results.emplace_back(insertOp.getDest());
    results.emplace_back(insertOp.getSource());
    return success();
  }

  return failure();
}

namespace {

/**
 * @brief Remove matching insert_slice-extract_slice pairs through commuting
 * disjoint tensor-chain operations.
 */
struct RemoveInsertSliceExtractSlicePair final
    : OpRewritePattern<ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractSliceOp extractSliceOp,
                                PatternRewriter& rewriter) const override {
    llvm::SmallVector<Operation*> traversedOps;
    Value currentTensor = extractSliceOp.getTensor();
    InsertSliceOp matchedInsertSliceOp = nullptr;

    while (auto* definingOp = currentTensor.getDefiningOp()) {
      if (!isTensorChainOp(definingOp)) {
        break;
      }

      if (auto insertSliceOp = llvm::dyn_cast<InsertSliceOp>(definingOp)) {
        const auto relation = classifyRanges(
            insertSliceOp.getOffset(), insertSliceOp.getSize(),
            extractSliceOp.getOffset(), extractSliceOp.getSize());
        if (relation == RangeRelation::Equal) {
          matchedInsertSliceOp = insertSliceOp;
          break;
        }
        if (relation != RangeRelation::Disjoint) {
          return failure();
        }
      } else if (auto insertOp = llvm::dyn_cast<InsertOp>(definingOp)) {
        if (classifyIndexAndRange(
                insertOp.getIndex(), extractSliceOp.getOffset(),
                extractSliceOp.getSize()) != RangeRelation::Disjoint) {
          return failure();
        }
      } else if (auto nestedExtractOp = llvm::dyn_cast<ExtractOp>(definingOp)) {
        if (classifyIndexAndRange(
                nestedExtractOp.getIndex(), extractSliceOp.getOffset(),
                extractSliceOp.getSize()) != RangeRelation::Disjoint) {
          return failure();
        }
      } else if (auto nestedExtractSliceOp =
                     llvm::dyn_cast<ExtractSliceOp>(definingOp)) {
        if (classifyRanges(
                nestedExtractSliceOp.getOffset(),
                nestedExtractSliceOp.getSize(), extractSliceOp.getOffset(),
                extractSliceOp.getSize()) != RangeRelation::Disjoint) {
          return failure();
        }
      }

      traversedOps.push_back(definingOp);
      currentTensor = getTensorChainInput(definingOp);
    }

    if (!matchedInsertSliceOp) {
      return failure();
    }

    Value outTensor = matchedInsertSliceOp.getDest();
    if (!traversedOps.empty()) {
      Operation* oldestCommutedOp = traversedOps.back();
      rewriter.modifyOpInPlace(oldestCommutedOp, [&]() {
        setTensorChainInput(oldestCommutedOp, matchedInsertSliceOp.getDest());
      });
      outTensor = getTensorChainOutput(traversedOps.front());
      if (!outTensor) {
        return failure();
      }
    }

    rewriter.replaceOp(extractSliceOp,
                       {outTensor, matchedInsertSliceOp.getSource()});
    return success();
  }
};

} // namespace

void ExtractSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                 MLIRContext* context) {
  results.add<RemoveInsertSliceExtractSlicePair>(context);
}
