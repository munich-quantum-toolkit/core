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
 * @brief Determine whether a qtensor.insert / qtensor.extract pair can be
 * safely removed without violating linearity.
 *
 * @param insertOp The insert operation to test.
 * @param extractOp The extract operation to test.
 * @return `true` if the insert's scalar is the same value produced by the
 * extract and their indices are equivalent, `false` otherwise.
 */
static bool isRemovableExtractInsertPair(InsertOp insertOp,
                                         ExtractOp extractOp) {
  return insertOp.getScalar() == extractOp.getResult() &&
         areEquivalentIndices(insertOp.getIndex(), extractOp.getIndex());
}

/**
 * Locate the `qtensor.extract` that corresponds to the scalar being inserted by
 * the given `qtensor.insert` by walking the tensor's defining-value chain.
 *
 * Traversal stops and returns `nullptr` if the insert's index is not a constant
 * integer, if any intervening index encountered is not constant, if a later
 * insert writes to an equivalent index (shadowing the original), or if no
 * matching extract is found.
 *
 * @param insertOp The `qtensor.insert` operation to match.
 * @return ExtractOp The matching `qtensor.extract` operation, or `nullptr` if no
 * matching extract exists or if matching cannot be determined.
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

  /**
   * @brief Match and remove a removable extract–insert pair.
   *
   * Locates an extract operation that corresponds to the provided `InsertOp` and,
   * if the pair is safe to remove, replaces the `qtensor.insert` with its
   * destination tensor and the matched `qtensor.extract` with its source tensor
   * operand (dropping the extracted scalar).
   *
   * @param op The `qtensor.insert` operation to match and potentially rewrite.
   * @param rewriter Pattern rewriter used to perform replacements.
   * @return LogicalResult `success()` if a matching removable pair was found and
   * the rewrites were applied, `failure()` otherwise.
   */
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

} /**
 * @brief Validate destination and index invariants for a `qtensor.insert` op.
 *
 * If the `index` operand is a constant integer, this verifies that the index is
 * greater than or equal to zero and, when the destination tensor's first
 * dimension is statically known, that the index is less than that dimension
 * size. No checks are performed when the index is not a compile-time constant.
 *
 * @return LogicalResult `success()` if checks pass; `failure()` and an emitted
 * op error if the constant index is negative or exceeds the destination
 * dimension.
 */

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

/**
 * @brief Register canonicalization patterns for this operation.
 *
 * Adds the RemoveExtractInsertPair rewrite pattern into `results` so the
 * canonicalizer can simplify removable `qtensor.insert`/`qtensor.extract`
 * pairs.
 *
 * @param results Pattern list to populate with canonicalization patterns.
 * @param context MLIR context used to construct the pattern.
 */
void InsertOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<RemoveExtractInsertPair>(context);
}
