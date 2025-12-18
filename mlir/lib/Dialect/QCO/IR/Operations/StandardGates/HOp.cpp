/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/QCOUtils.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

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
