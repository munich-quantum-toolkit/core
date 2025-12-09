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
 * @brief Merge subsequent RXX operations on the same qubits by adding their
 * angles.
 */
struct MergeSubsequentRXX final : OpRewritePattern<RXXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RXXOp op,
                                PatternRewriter& rewriter) const override {
    return mergeTwoTargetOneParameter(op, rewriter);
  }
};

/**
 * @brief Remove trivial RXX operations.
 */
struct RemoveTrivialRXX final : OpRewritePattern<RXXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RXXOp op,
                                PatternRewriter& rewriter) const override {
    return removeTrivialTwoTargetOneParameter(op, rewriter);
  }
};

} // namespace

void RXXOp::build(OpBuilder& builder, OperationState& state, Value qubit0In,
                  Value qubit1In, const std::variant<double, Value>& theta) {
  const auto& thetaOperand = variantToValue(builder, state, theta);
  build(builder, state, qubit0In, qubit1In, thetaOperand);
}

void RXXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<MergeSubsequentRXX, RemoveTrivialRXX>(context);
}
