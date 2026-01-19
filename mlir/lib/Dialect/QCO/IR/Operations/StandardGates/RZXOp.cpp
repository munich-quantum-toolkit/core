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
 * @brief Merge subsequent RZX operations on the same qubits by adding their
 * angles.
 */
struct MergeSubsequentRZX final : OpRewritePattern<RZXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RZXOp op,
                                PatternRewriter& rewriter) const override {
    return mergeTwoTargetOneParameter(op, rewriter);
  }
};

/**
 * @brief Remove trivial RZX operations.
 */
struct RemoveTrivialRZX final : OpRewritePattern<RZXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RZXOp op,
                                PatternRewriter& rewriter) const override {
    return removeTrivialTwoTargetOneParameter(op, rewriter);
  }
};

} // namespace

void RZXOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  Value qubit0In, Value qubit1In,
                  const std::variant<double, Value>& theta) {
  auto thetaOperand = variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand);
}

void RZXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<MergeSubsequentRZX, RemoveTrivialRZX>(context);
}
