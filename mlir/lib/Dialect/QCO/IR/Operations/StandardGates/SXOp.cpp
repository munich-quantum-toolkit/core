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
#include "mlir/Dialect/QCO/Utils/UnitaryMatrix.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include <complex>

using namespace mlir;
using namespace mlir::qco;

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

Matrix2x2 SXOp::getUnitaryMatrix() {
  constexpr auto m00 = std::complex<double>{0.5, 0.5};
  constexpr auto m01 = std::complex<double>{0.5, -0.5};
  return Matrix2x2::fromElements(m00, m01,  // row 0
                                 m01, m00); // row 1
}
