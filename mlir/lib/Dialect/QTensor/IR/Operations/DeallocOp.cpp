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

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qtensor;

namespace {

/**
 * @brief Remove matching allocation-deallocation pairs without operations
 * between them.
 */
struct RemoveAllocDeallocPair final : OpRewritePattern<DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  /**
   * @brief Removes a paired qtensor::AllocOp when the given DeallocOp's tensor
   * is directly defined by that AllocOp.
   *
   * @param op The DeallocOp to match; its tensor operand is checked for a
   * defining qtensor::AllocOp.
   * @param rewriter The PatternRewriter used to erase matching operations.
   * @return LogicalResult `success()` if a defining AllocOp was found and both
   *         the AllocOp and DeallocOp were erased, `failure()` otherwise.
   */
  LogicalResult matchAndRewrite(DeallocOp op,
                                PatternRewriter& rewriter) const override {
    // Check whether the tensor is directly defined by a qtensor::AllocOp.
    auto tensor = op.getTensor();
    auto allocOp = tensor.getDefiningOp<AllocOp>();
    if (!allocOp) {
      return failure();
    }

    // Remove the AllocOp and the DeallocOp.
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
