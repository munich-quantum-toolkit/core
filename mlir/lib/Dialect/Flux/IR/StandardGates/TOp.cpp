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

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::flux;
using namespace mlir::utils;

namespace {

/**
 * @brief Remove T operations that immediately follow Tdg operations.
 */
struct RemoveTAfterTdg final : OpRewritePattern<TOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePair<TdgOp>(op, rewriter);
  }
};

} // namespace

DenseElementsAttr TOp::tryGetStaticMatrix() { return getMatrixT(getContext()); }

void TOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<RemoveTAfterTdg>(context);
}
