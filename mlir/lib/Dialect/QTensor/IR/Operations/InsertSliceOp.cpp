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
#include "mlir/Dialect/QTensor/IR/QTensorUtils.h"

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qtensor;

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
          AccessRelation::Disjoint) {
        return nullptr;
      }
      current = nestedInsertOp.getDest();
      continue;
    }
    if (auto nestedInsertSliceOp = llvm::dyn_cast<InsertSliceOp>(definingOp)) {
      if (classifyRanges(nestedInsertSliceOp.getOffset(),
                         nestedInsertSliceOp.getSize(), offset,
                         size) != AccessRelation::Disjoint) {
        return nullptr;
      }
      current = nestedInsertSliceOp.getDest();
      continue;
    }
    if (auto extractOp = llvm::dyn_cast<ExtractOp>(definingOp)) {
      if (classifyIndexAndRange(extractOp.getIndex(), offset, size) !=
          AccessRelation::Disjoint) {
        return nullptr;
      }
      current = extractOp.getTensor();
      continue;
    }
    if (auto extractSliceOp = llvm::dyn_cast<ExtractSliceOp>(definingOp)) {
      const auto relation = classifyRanges(
          extractSliceOp.getOffset(), extractSliceOp.getSize(), offset, size);
      if (relation == AccessRelation::Equal) {
        return extractSliceOp;
      }
      if (relation != AccessRelation::Disjoint) {
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

  if (!areEquivalentRanges(insertSliceOp.getOffset(), insertSliceOp.getSize(),
                           extractSliceOp.getOffset(),
                           extractSliceOp.getSize())) {
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
