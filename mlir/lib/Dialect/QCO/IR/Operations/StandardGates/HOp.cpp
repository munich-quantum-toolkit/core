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
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <numbers>

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Remove subsequent H operations on the same qubit.
 */
struct RemoveSubsequentH final : OpRewritePattern<HOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(HOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePairOneTargetZeroParameter<HOp>(op, rewriter);
  }
};

} // namespace

void HOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<RemoveSubsequentH>(context);
}

Eigen::Matrix2cd HOp::getUnitaryMatrix() {
  constexpr auto x = 1.0 / std::numbers::sqrt2;
  return Eigen::Matrix2cd{{x, x}, {x, -1.0 * x}};
}
