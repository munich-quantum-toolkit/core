/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::qtensor;

namespace {

/// Remove matching allocation-deallocation pairs without operations
/// between them.
struct RemoveAllocDeallocPair final : OpRewritePattern<DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeallocOp op,
                                PatternRewriter& rewriter) const override {
    // Check whether the tensor is directly defined by an otherwise unused
    // qtensor::AllocOp.
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
  results.add(+[](DeallocOp op, PatternRewriter& rewriter) {
    auto insert = op.getTensor().getDefiningOp<InsertOp>();
    if (!insert || insert->getBlock() != op->getBlock() ||
        !insert.getScalar().getDefiningOp<qco::MeasureOp>()) {
      return failure();
    }
    // Expose discarded measurement outputs without changing unmeasured wires.
    qco::SinkOp::create(rewriter, op.getLoc(), insert.getScalar());
    rewriter.replaceOp(insert, insert.getDest());
    return success();
  });
}
