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
 * @brief Remove Tdg operations that immediately follow T operations.
 */
struct RemoveTdgAfterT final : OpRewritePattern<TdgOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TdgOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePairOneTargetZeroParameter<TOp>(op, rewriter);
  }
};

/**
 * @brief Merge subsequent Tdg operations on the same qubit into an Sdg
 * operation.
 */
struct MergeSubsequentTdg final : OpRewritePattern<TdgOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TdgOp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetZeroParameter<SdgOp>(op, rewriter);
  }
};

} // namespace

void TdgOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<RemoveTdgAfterT, MergeSubsequentTdg>(context);
}

Matrix2x2 TdgOp::getUnitaryMatrix() {
  const auto m11 = std::polar(1.0, -std::numbers::pi / 4);
  return Matrix2x2::fromElements(1, 0,    // row 0
                                 0, m11); // row 1
}
