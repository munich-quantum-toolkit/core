/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Flux/FluxUtils.h"
#include "mlir/Dialect/Flux/IR/FluxDialect.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::flux;

namespace {

/**
 * @brief Remove SX operations that immediately follow SXdg operations.
 */
struct RemoveSXAfterSXdg final : OpRewritePattern<SXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SXOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePairOneTargetZeroParameter<SXdgOp>(op, rewriter);
  }
};

/**
 * @brief Merge subsequent SX operations on the same qubit into an X operation.
 */
struct MergeSubsequentSX final : OpRewritePattern<SXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SXOp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetZeroParameter<XOp>(op, rewriter);
  }
};

} // namespace

void SXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<RemoveSXAfterSXdg, MergeSubsequentSX>(context);
}
