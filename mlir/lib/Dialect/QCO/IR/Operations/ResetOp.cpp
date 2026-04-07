/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorUtils.h"

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

/**
 * @brief Determine whether a `qtensor::ExtractOp` is provably sourced from a
 * `qtensor::AllocOp`.
 *
 * @param extractOp The extract operation whose tensor provenance to trace.
 * @return `true` if tracing the defining operations of the extract's tensor
 * reaches a `qtensor::AllocOp` while tolerating intervening
 * `qtensor::InsertOp`/`qtensor::ExtractOp` only when those ops use constant
 * indices that are not equivalent to the extract's index; `false` otherwise.
 * Non-constant indices cause this check to fail (`false`).
 */
static bool originatesFromQTensorAlloc(qtensor::ExtractOp extractOp) {
  auto current = extractOp.getTensor();

  auto extractIndex = extractOp.getIndex();
  if (!getConstantIntValue(extractIndex)) {
    return false;
  }

  while (auto* definingOp = current.getDefiningOp()) {
    if (llvm::isa<qtensor::AllocOp>(definingOp)) {
      return true;
    }

    if (auto nestedExtractOp = llvm::dyn_cast<qtensor::ExtractOp>(definingOp)) {
      auto nestedExtractIndex = nestedExtractOp.getIndex();
      if (!getConstantIntValue(nestedExtractIndex)) {
        return false;
      }
      if (qtensor::areEquivalentIndices(extractIndex, nestedExtractIndex)) {
        return false;
      }
      current = nestedExtractOp.getTensor();
      continue;
    }

    if (auto insertOp = llvm::dyn_cast<qtensor::InsertOp>(definingOp)) {
      auto insertIndex = insertOp.getIndex();
      if (!getConstantIntValue(insertIndex)) {
        return false;
      }
      if (qtensor::areEquivalentIndices(extractIndex, insertIndex)) {
        return false;
      }
      current = insertOp.getDest();
      continue;
    }

    return false;
  }

  return false;
}

namespace {

/**
 * @brief Remove reset operations that immediately follow a `qtensor.extract`
 * operation.
 */
struct RemoveResetAfterExtract final : OpRewritePattern<ResetOp> {
  using OpRewritePattern::OpRewritePattern;

  /**
   * Rewrites a `qtensor::ResetOp` away when its qubit input can be proven to
   * originate from a `qtensor::AllocOp` via an extract/insert provenance chain.
   *
   * If the qubit operand is defined by a `qtensor::ExtractOp` and that extract
   * traces back to an `qtensor::AllocOp` (subject to the provenance rules), the
   * pattern replaces the `ResetOp` with the qubit operand.
   *
   * @returns `success` if the `ResetOp` was replaced and removed, `failure`
   * otherwise.
   */
  LogicalResult matchAndRewrite(ResetOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is an ExtractOp
    auto extractOp = op.getQubitIn().getDefiningOp<qtensor::ExtractOp>();
    if (!extractOp) {
      return failure();
    }

    // Check if the tensor originates from an AllocOp
    if (!originatesFromQTensorAlloc(extractOp)) {
      return failure();
    }

    // Remove the ResetOp
    rewriter.replaceOp(op, op.getQubitIn());
    return success();
  }
};

} // namespace

OpFoldResult ResetOp::fold(FoldAdaptor /*adaptor*/) {
  if (getQubitIn().getDefiningOp<AllocOp>()) {
    return getQubitIn();
  }

  return {};
}

/**
 * @brief Register canonicalization patterns for ResetOp.
 *
 * Adds the RemoveResetAfterExtract rewrite pattern to the provided pattern
 * set so ResetOp instances can be canonicalized based on extract-origin
 * provenance.
 *
 * @param results Pattern set to populate with canonicalization patterns.
 * @param context MLIR context used to construct the rewrite pattern.
 */
void ResetOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add<RemoveResetAfterExtract>(context);
}
