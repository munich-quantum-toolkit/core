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
#include "mlir/Dialect/QCO/Utils/UnitaryMatrix.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Remove identity operations.
 */
struct RemoveId final : OpRewritePattern<IdOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IdOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, op.getQubitIn());
    return success();
  }
};

} // namespace

void IdOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<RemoveId>(context);
}

Matrix2x2 IdOp::getUnitaryMatrix() {
  return Matrix2x2::fromElements(1, 0,  // row 0
                                 0, 1); // row 1
}
