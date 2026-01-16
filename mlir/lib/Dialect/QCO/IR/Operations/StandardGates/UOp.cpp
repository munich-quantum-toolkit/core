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
#include "mlir/Dialect/Utils/Utils.h"

#include <cmath>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <numbers>
#include <variant>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::utils;

namespace {

/**
 * @brief Replace U(0, 0, lambda) with P(lambda).
 */
struct ReplaceUWithP final : OpRewritePattern<UOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UOp op,
                                PatternRewriter& rewriter) const override {
    const auto theta = valueToDouble(op.getTheta());
    const auto phi = valueToDouble(op.getPhi());
    if (!theta || std::abs(*theta) > TOLERANCE || !phi ||
        std::abs(*phi) > TOLERANCE) {
      return failure();
    }

    auto pOp =
        rewriter.create<POp>(op.getLoc(), op.getInputQubit(0), op.getLambda());
    rewriter.replaceOp(op, pOp.getResult());

    return success();
  }
};

/**
 * @brief Replace U(theta, -pi / 2, pi / 2) with RX(theta).
 */
struct ReplaceUWithRX final : OpRewritePattern<UOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UOp op,
                                PatternRewriter& rewriter) const override {
    const auto phi = valueToDouble(op.getPhi());
    const auto lambda = valueToDouble(op.getLambda());
    if (!phi || std::abs(*phi + (std::numbers::pi / 2.0)) > TOLERANCE ||
        !lambda || std::abs(*lambda - (std::numbers::pi / 2.0)) > TOLERANCE) {
      return failure();
    }

    auto rxOp =
        rewriter.create<RXOp>(op.getLoc(), op.getInputQubit(0), op.getTheta());
    rewriter.replaceOp(op, rxOp.getResult());

    return success();
  }
};

/**
 * @brief Replace U(theta, 0, 0) with RY(theta).
 */
struct ReplaceUWithRY final : OpRewritePattern<UOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UOp op,
                                PatternRewriter& rewriter) const override {
    const auto phi = valueToDouble(op.getPhi());
    const auto lambda = valueToDouble(op.getLambda());
    if (!phi || std::abs(*phi) > TOLERANCE || !lambda ||
        std::abs(*lambda) > TOLERANCE) {
      return failure();
    }

    auto ryOp =
        rewriter.create<RYOp>(op.getLoc(), op.getInputQubit(0), op.getTheta());
    rewriter.replaceOp(op, ryOp.getResult());

    return success();
  }
};

} // namespace

void UOp::build(OpBuilder& builder, OperationState& state, Value qubitIn,
                const std::variant<double, Value>& theta,
                const std::variant<double, Value>& phi,
                const std::variant<double, Value>& lambda) {
  auto thetaOperand = variantToValue(builder, state.location, theta);
  auto phiOperand = variantToValue(builder, state.location, phi);
  auto lambdaOperand = variantToValue(builder, state.location, lambda);
  build(builder, state, qubitIn, thetaOperand, phiOperand, lambdaOperand);
}

void UOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<ReplaceUWithP, ReplaceUWithRX, ReplaceUWithRY>(context);
}
