/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/QCOUtils.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Remove S operations that immediately follow Sdg operations.
 */
struct RemoveSAfterSdg final : OpRewritePattern<SOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePairOneTargetZeroParameter<SdgOp>(op, rewriter);
  }
};

/**
 * @brief Merge subsequent S operations on the same qubit into a Z operation.
 */
struct MergeSubsequentS final : OpRewritePattern<SOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SOp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetZeroParameter<ZOp>(op, rewriter);
  }
};

} // namespace

void SOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<RemoveSAfterSdg, MergeSubsequentS>(context);
}
