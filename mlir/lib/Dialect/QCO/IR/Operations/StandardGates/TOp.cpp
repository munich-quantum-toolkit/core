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
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include <complex>
#include <numbers>

using namespace mlir;
using namespace mlir::qco;

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

/**
 * @brief Merge T operations separated only by `ctrl` hops on a control wire
 * into an S operation.
 */
struct MergeTThroughCtrlControlChain final : OpRewritePattern<TOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TOp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetZeroParameterThroughCtrlControlChain<SOp>(op,
                                                                   rewriter);
  }
};

} // namespace

void TOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<RemoveTAfterTdg, MergeSubsequentT, MergeTThroughCtrlControlChain>(
      context);
}

Matrix2x2 TOp::getUnitaryMatrix() {
  const auto m11 = std::polar(1.0, std::numbers::pi / 4);
  return Matrix2x2::fromElements(1, 0,    // row 0
                                 0, m11); // row 1
}
