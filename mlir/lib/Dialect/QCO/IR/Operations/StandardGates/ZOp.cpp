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

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Remove subsequent Z operations on the same qubit.
 */
struct RemoveSubsequentZ final : OpRewritePattern<ZOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ZOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePairOneTargetZeroParameter<ZOp>(op, rewriter);
  }
};

} // namespace

void ZOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<RemoveSubsequentZ>(context);
}
