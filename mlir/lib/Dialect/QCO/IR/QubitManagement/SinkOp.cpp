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

#include <llvm/Support/Casting.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Remove matching allocation/static and sink pairs without operations
 * between them.
 */
struct RemoveAllocSinkPair final : OpRewritePattern<SinkOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SinkOp op,
                                PatternRewriter& rewriter) const override {
    auto* defOp = op.getQubit().getDefiningOp();
    if (!llvm::isa<AllocOp, StaticOp>(defOp)) {
      return failure();
    }

    rewriter.eraseOp(op);
    rewriter.eraseOp(defOp);
    return success();
  }
};

} // namespace

void SinkOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<RemoveAllocSinkPair>(context);
}
