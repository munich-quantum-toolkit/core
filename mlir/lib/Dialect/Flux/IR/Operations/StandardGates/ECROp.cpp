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
 * @brief Remove subsequent ECR operations on the same qubits.
 */
struct RemoveSubsequentECR final : OpRewritePattern<ECROp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ECROp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePairTwoTargetZeroParameter<ECROp>(op, rewriter);
  }
};
} // namespace

DenseElementsAttr ECROp::tryGetStaticMatrix() {
  return getMatrixECR(getContext());
}

void ECROp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<RemoveSubsequentECR>(context);
}
