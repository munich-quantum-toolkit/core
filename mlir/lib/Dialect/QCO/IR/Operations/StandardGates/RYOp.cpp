/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <variant>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::utils;

namespace {

/**
 * @brief Merge subsequent RY operations on the same qubit by adding their
 * angles.
 */
struct MergeSubsequentRY final : OpRewritePattern<RYOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RYOp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetOneParameter(op, rewriter);
  }
};

/**
 * @brief Remove trivial RY operations.
 */
struct RemoveTrivialRY final : OpRewritePattern<RYOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RYOp op,
                                PatternRewriter& rewriter) const override {
    return removeTrivialOneTargetOneParameter(op, rewriter);
  }
};

} // namespace

void RYOp::build(OpBuilder& odsBuilder, OperationState& odsState, Value qubitIn,
                 const std::variant<double, Value>& theta) {
  auto thetaOperand = variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, qubitIn, thetaOperand);
}

void RYOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<MergeSubsequentRY, RemoveTrivialRY>(context);
}
