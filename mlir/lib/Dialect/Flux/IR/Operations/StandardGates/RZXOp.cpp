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

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
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

void RZXOp::build(OpBuilder& builder, OperationState& state,
                  const Value qubit0In, const Value qubit1In,
                  const std::variant<double, Value>& theta) {
  const auto& thetaOperand = variantToValue(builder, state, theta);
  build(builder, state, qubit0In, qubit1In, thetaOperand);
}

void RZXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<MergeSubsequentRZX, RemoveTrivialRZX>(context);
}
