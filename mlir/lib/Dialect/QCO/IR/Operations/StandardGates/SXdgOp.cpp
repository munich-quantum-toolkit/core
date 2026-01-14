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

#include <complex>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Remove SXdg operations that immediately follow SX operations.
 */
struct RemoveSXdgAfterSX final : OpRewritePattern<SXdgOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SXdgOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePairOneTargetZeroParameter<SXOp>(op, rewriter);
  }
};

/**
 * @brief Merge subsequent SXdg operations on the same qubit into an X
 * operation.
 */
struct MergeSubsequentSXdg final : OpRewritePattern<SXdgOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SXdgOp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetZeroParameter<XOp>(op, rewriter);
  }
};

} // namespace

void SXdgOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<RemoveSXdgAfterSX, MergeSubsequentSXdg>(context);
}

Eigen::Matrix2cd SXdgOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  const auto m00 = (1.0 - 1i) / 2.0;
  const auto m01 = (1.0 + 1i) / 2.0;
  return Eigen::Matrix2cd{{m00, m01}, {m01, m00}};
}
