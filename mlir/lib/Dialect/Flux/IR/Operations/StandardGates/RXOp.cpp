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
#include "mlir/Dialect/Utils/Utils.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <variant>

using namespace mlir;
using namespace mlir::flux;
using namespace mlir::utils;

namespace {

/**
 * @brief Merge subsequent RX operations on the same qubit by adding their
 * angles.
 */
struct MergeSubsequentRX final : OpRewritePattern<RXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RXOp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetOneParameter(op, rewriter);
  }
};

/**
 * @brief Remove trivial RX operations.
 */
struct RemoveTrivialRX final : OpRewritePattern<RXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RXOp op,
                                PatternRewriter& rewriter) const override {
    return removeTrivialOneTargetOneParameter(op, rewriter);
  }
};

} // namespace

void RXOp::build(OpBuilder& builder, OperationState& state, Value qubitIn,
                 const std::variant<double, Value>& theta) {
  const auto& thetaOperand = variantToValue(builder, state, theta);
  build(builder, state, qubitIn, thetaOperand);
}

void RXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<MergeSubsequentRX, RemoveTrivialRX>(context);
}
