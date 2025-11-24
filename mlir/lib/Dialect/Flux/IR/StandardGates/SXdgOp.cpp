/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Flux/FluxUtils.h"
#include "mlir/Dialect/Flux/IR/FluxDialect.h"
#include "mlir/Dialect/Utils/MatrixUtils.h"

#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::flux;
using namespace mlir::utils;

namespace {

/**
 * @brief Remove SXdg operations that immediately follow SX operations.
 */
struct RemoveSXdgAfterSX final : OpRewritePattern<SXdgOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SXdgOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePair<SXOp>(op, rewriter);
  }
};

} // namespace

DenseElementsAttr SXdgOp::tryGetStaticMatrix() {
  return getMatrixSXdg(getContext());
}

void SXdgOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<RemoveSXdgAfterSX>(context);
}
