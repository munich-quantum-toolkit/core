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
 * @brief Finds the `qtensor.extract` operation corresponding to a given
 * `qtensor.insert` operation.
 *
 * @details The function traverses the tensor chain of the `qtensor.insert`
 * operation until it finds the matching `qtensor.extract` operation.
 */
static ExtractOp findMatchingExtractInTensorChain(InsertOp insertOp) {
  auto current = insertOp.getDest();
  auto insertIndex = insertOp.getIndex();

  if (!getConstantIntValue(insertIndex)) {
    return nullptr;
  }

  while (auto* definingOp = current.getDefiningOp()) {
    if (auto nestedInsertOp = llvm::dyn_cast<InsertOp>(definingOp)) {
      auto nestedInsertIndex = nestedInsertOp.getIndex();
      if (!getConstantIntValue(nestedInsertIndex)) {
        return nullptr;
      }
      // A more recent write to the same index shadows all older extracts
      if (areEquivalentIndices(nestedInsertIndex, insertIndex)) {
        return nullptr;
      }
      current = nestedInsertOp.getDest();
      continue;
    }
    if (auto extractOp = llvm::dyn_cast<ExtractOp>(definingOp)) {
      auto extractIndex = extractOp.getIndex();
      if (!getConstantIntValue(extractIndex)) {
        return nullptr;
      }
      if (areEquivalentIndices(extractIndex, insertIndex)) {
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
 * @brief Remove matching extract-insert pairs.
 */
struct RemoveExtractInsertPair final : OpRewritePattern<InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp op,
                                PatternRewriter& rewriter) const override {
    auto extractOp = findMatchingExtractInTensorChain(op);
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
