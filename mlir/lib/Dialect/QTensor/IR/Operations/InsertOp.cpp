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
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qtensor;

/**
 * @brief Checks whether removing an extract-insert pair is linearity-safe.
 */
static bool isRemovableExtractInsertPair(InsertOp insertOp,
                                         ExtractOp extractOp) {
  return insertOp.getScalar() == extractOp.getResult() &&
         areEquivalentIndices(insertOp.getIndex(), extractOp.getIndex());
}

/**
 * @brief Fold the direct pattern
 * `insert(extract(tensor, idx).qubit, extract(tensor, idx).out, idx)`.
 */
static Value foldInsertAfterExtract(InsertOp insertOp) {
  auto extractOp = insertOp.getScalar().getDefiningOp<ExtractOp>();
  if (!extractOp) {
    return nullptr;
  }

  if (insertOp.getDest() != extractOp.getOutTensor()) {
    return nullptr;
  }

  if (!isRemovableExtractInsertPair(insertOp, extractOp)) {
    return nullptr;
  }

  return extractOp.getTensor();
}

OpFoldResult InsertOp::fold(FoldAdaptor /*adaptor*/) {
  if (auto result = foldInsertAfterExtract(*this)) {
    return result;
  }
  return {};
}

/**
 * @brief Find a matching `qtensor.extract` for an insert index in a tensor
 * chain by traversing nested scalar tensor ops.
 */
static ExtractOp findMatchingExtractInTensorChain(Value tensor, Value index) {
  Value current = tensor;
  while (Operation* definingOp = current.getDefiningOp()) {
    if (auto nestedInsertOp = llvm::dyn_cast<InsertOp>(definingOp)) {
      // A more recent write to the same index shadows all older extracts.
      if (areEquivalentIndices(nestedInsertOp.getIndex(), index)) {
        return nullptr;
      }
      current = nestedInsertOp.getDest();
      continue;
    }
    if (auto extractOp = llvm::dyn_cast<ExtractOp>(definingOp)) {
      if (areEquivalentIndices(extractOp.getIndex(), index)) {
        return extractOp;
      }
      current = extractOp.getTensor();
      continue;
    }
    break;
  }
  return nullptr;
}

namespace {

/**
 * @brief Remove matching `qtensor.insert` and `qtensor.extract` pairs.
 */
struct RemoveExtractInsertPair final : OpRewritePattern<InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp op,
                                PatternRewriter& rewriter) const override {
    auto extractOp =
        findMatchingExtractInTensorChain(op.getDest(), op.getIndex());
    if (!extractOp) {
      return failure();
    }

    if (!isRemovableExtractInsertPair(op, extractOp)) {
      return failure();
    }

    rewriter.replaceOp(op, op.getDest());
    rewriter.replaceOp(extractOp, {extractOp.getTensor(), nullptr});

    return success();
  }
};

} // namespace

LogicalResult InsertOp::verify() {
  auto dstDim = getDest().getType().getDimSize(0);
  auto index = getConstantIntValue(getIndex());

  if (index) {
    if (*index < 0) {
      return emitOpError("Index must be non-negative");
    }
    if (!ShapedType::isDynamic(dstDim) && *index >= dstDim) {
      return emitOpError("Index exceeds tensor dimension");
    }
  }

  return success();
}

void InsertOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<RemoveExtractInsertPair>(context);
}
