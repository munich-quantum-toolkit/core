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

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Remove reset operations that immediately follow a `qco.alloc`
 * operation.
 */
struct RemoveResetAfterAlloc final : OpRewritePattern<ResetOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ResetOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is an AllocOp
    if (auto allocOp = op.getQubitIn().getDefiningOp<AllocOp>(); !allocOp) {
      return failure();
    }

    // Remove the ResetOp
    rewriter.replaceOp(op, op.getQubitIn());
    return success();
  }
};

/**
 * @brief Remove reset operations that immediately follow a `qtensor.extract`
 * operation.
 */
struct RemoveResetAfterExtract final : OpRewritePattern<ResetOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ResetOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is an ExtractOp
    if (auto extractOp = op.getQubitIn().getDefiningOp<qtensor::ExtractOp>();
        !extractOp) {
      return failure();
    }

    // Remove the ResetOp
    rewriter.replaceOp(op, op.getQubitIn());
    return success();
  }
};

} // namespace

void ResetOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add<RemoveResetAfterAlloc, RemoveResetAfterExtract>(context);
}
