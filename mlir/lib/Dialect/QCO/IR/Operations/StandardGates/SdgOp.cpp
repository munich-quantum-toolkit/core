/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/QCOUtils.h"

#include <Eigen/Core>
#include <complex>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

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

Eigen::Matrix2cd SdgOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  return Eigen::Matrix2cd{{1.0, 0.0},  // row 0
                          {0.0, -1i}}; // row 1
}
