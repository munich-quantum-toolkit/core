/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/Support/Casting.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qtensor;

namespace {

/**
 * @brief Remove matching allocation and deallocation pairs without operations
 * between them.
 */
struct RemoveAllocDeallocPair final : OpRewritePattern<DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeallocOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is an qtensor::AllocOp
    auto* defOp = op.getTensor().getDefiningOp();
    if (!llvm::isa<AllocOp>(defOp)) {
      return failure();
    }

    // Remove the AllocOp and the DeallocOp
    rewriter.eraseOp(op);
    rewriter.eraseOp(defOp);
    return success();
  }
};

} // namespace

void DeallocOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<RemoveAllocDeallocPair>(context);
}
