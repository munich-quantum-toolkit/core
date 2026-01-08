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

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Remove matching allocation and deallocation pairs without operations
 * between them.
 */
struct RemoveAllocDeallocPair final : OpRewritePattern<DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeallocOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is an AllocOp
    auto allocOp = op.getQubit().getDefiningOp<AllocOp>();
    if (!allocOp) {
      return failure();
    }

    // Remove the AllocOp and the DeallocOp
    rewriter.eraseOp(op);
    rewriter.eraseOp(allocOp);
    return success();
  }
};

} // namespace

void DeallocOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<RemoveAllocDeallocPair>(context);
}
