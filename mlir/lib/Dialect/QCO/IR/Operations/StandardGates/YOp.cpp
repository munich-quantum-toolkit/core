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
 * @brief Remove subsequent Y operations on the same qubit.
 */
struct RemoveSubsequentY final : OpRewritePattern<YOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(YOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePairOneTargetZeroParameter<YOp>(op, rewriter);
  }
};

} // namespace

void YOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<RemoveSubsequentY>(context);
}

Eigen::Matrix2cd YOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  return Eigen::Matrix2cd{{0, -1i}, {1i, 0}};
}
