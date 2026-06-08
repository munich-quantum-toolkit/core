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

Matrix2 SOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  return Matrix2::fromElements(1.0, 0.0, 0.0, 1i);
}
