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
 * @brief Remove Sdg operations that immediately follow S operations.
 */
struct RemoveSdgAfterS final : OpRewritePattern<SdgOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SdgOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePairOneTargetZeroParameter<SOp>(op, rewriter);
  }
};

/**
 * @brief Merge subsequent Sdg operations on the same qubit into a Z operation.
 */
struct MergeSubsequentSdg final : OpRewritePattern<SdgOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SdgOp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetZeroParameter<ZOp>(op, rewriter);
  }
};

} // namespace

void SdgOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<RemoveSdgAfterS, MergeSubsequentSdg>(context);
}
