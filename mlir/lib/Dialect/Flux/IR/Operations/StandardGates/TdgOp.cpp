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
 * @brief Remove Tdg operations that immediately follow T operations.
 */
struct RemoveTdgAfterT final : OpRewritePattern<TdgOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TdgOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePairOneTargetZeroParameter<TOp>(op, rewriter);
  }
};

/**
 * @brief Merge subsequent Tdg operations on the same qubit into an Sdg
 * operation.
 */
struct MergeSubsequentTdg final : OpRewritePattern<TdgOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TdgOp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetZeroParameter<SdgOp>(op, rewriter);
  }
};

} // namespace

DenseElementsAttr TdgOp::tryGetStaticMatrix() {
  return getMatrixTdg(getContext());
}

void TdgOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<RemoveTdgAfterT, MergeSubsequentTdg>(context);
}
