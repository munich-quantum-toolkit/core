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

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Remove a DCX operation followed by a DCX operation with swapped
 *        targets.
 */
struct RemoveInversePairDCX final : OpRewritePattern<DCXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DCXOp op,
                                PatternRewriter& rewriter) const override {
    return removeTwoTargetZeroParameterPairWithSwappedTargets<DCXOp>(op,
                                                                     rewriter);
  }
};

} // namespace

void DCXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<RemoveInversePairDCX>(context);
}

Eigen::Matrix4cd DCXOp::getUnitaryMatrix() {
  return Eigen::Matrix4cd{{1, 0, 0, 0},  // row 0
                          {0, 0, 1, 0},  // row 1
                          {0, 0, 0, 1},  // row 2
                          {0, 1, 0, 0}}; // row 3
}
