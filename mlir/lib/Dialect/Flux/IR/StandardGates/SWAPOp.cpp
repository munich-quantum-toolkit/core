/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Flux/IR/FluxDialect.h"
#include "mlir/Dialect/Utils/MatrixUtils.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::flux;
using namespace mlir::utils;

/**
 * @brief Remove subsequent SWAP operations on the same qubit.
 */
struct RemoveSubsequentSWAP final : OpRewritePattern<SWAPOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SWAPOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is a SWAP operation
    auto prevOp = op.getQubit0In().getDefiningOp<SWAPOp>();
    if (!prevOp) {
      return failure();
    }
    if (op.getQubit1In() != prevOp.getQubit1Out()) {
      return failure();
    }

    // Remove both SWAP operations
    rewriter.replaceOp(prevOp, {prevOp.getQubit0In(), prevOp.getQubit1In()});
    rewriter.replaceOp(op, {op.getQubit0In(), op.getQubit1In()});

    return success();
  }
};

DenseElementsAttr SWAPOp::tryGetStaticMatrix() {
  return getMatrixSWAP(getContext());
}

void SWAPOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<RemoveSubsequentSWAP>(context);
}
