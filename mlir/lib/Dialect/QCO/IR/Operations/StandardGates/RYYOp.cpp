/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
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
 * @brief Merge subsequent RYY operations on the same qubits by adding their
 * angles.
 */
struct MergeSubsequentRYY final : OpRewritePattern<RYYOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RYYOp op,
                                PatternRewriter& rewriter) const override {
    return mergeTwoTargetOneParameter(op, rewriter);
  }
};

/**
 * @brief Remove trivial RYY operations.
 */
struct RemoveTrivialRYY final : OpRewritePattern<RYYOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RYYOp op,
                                PatternRewriter& rewriter) const override {
    return removeTrivialTwoTargetOneParameter(op, rewriter);
  }
};

} // namespace

void RYYOp::build(OpBuilder& builder, OperationState& state, Value qubit0In,
                  Value qubit1In, const std::variant<double, Value>& theta) {
  auto thetaOperand = variantToValue(builder, state, theta);
  build(builder, state, qubit0In, qubit1In, thetaOperand);
}

void RYYOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<MergeSubsequentRYY, RemoveTrivialRYY>(context);
}
