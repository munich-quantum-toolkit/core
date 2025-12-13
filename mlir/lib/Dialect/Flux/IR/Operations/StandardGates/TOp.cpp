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
 * @brief Remove T operations that immediately follow Tdg operations.
 */
struct RemoveTAfterTdg final : OpRewritePattern<TOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePairOneTargetZeroParameter<TdgOp>(op, rewriter);
  }
};

/**
 * @brief Merge subsequent T operations on the same qubit into an S operation.
 */
struct MergeSubsequentT final : OpRewritePattern<TOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TOp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetZeroParameter<SOp>(op, rewriter);
  }
};

} // namespace

void TOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<RemoveTAfterTdg, MergeSubsequentT>(context);
}
