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

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

namespace {

struct RemoveThreeBackToBackTripleOps final : OpRewritePattern<TripleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TripleOp op,
                                PatternRewriter& rewriter) const override {
    // TODO: Task 3
    auto prevOp1 = op.getQubitIn().getDefiningOp<TripleOp>();
    if (!prevOp1) {
      return failure();
    }

    auto prevOp2 = prevOp1.getQubitIn().getDefiningOp<TripleOp>();
    if (!prevOp2) {
      return failure();
    }

    rewriter.replaceOp(op, prevOp2.getQubitIn());
    return success();
    // llvm::reportFatalInternalError("Not implemented yet");
  }
};

} // namespace

void TripleOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<RemoveThreeBackToBackTripleOps>(context);
}
