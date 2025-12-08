/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Flux/IR/FluxDialect.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <numbers>
#include <variant>

using namespace mlir;
using namespace mlir::flux;
using namespace mlir::utils;

namespace {

/**
 * @brief Replace U(0, 0, lambda) with P(lambda).
 */
struct ReplaceUWithP final : OpRewritePattern<UOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UOp op,
                                PatternRewriter& rewriter) const override {
    const auto theta = UOp::getStaticParameter(op.getTheta());
    const auto phi = UOp::getStaticParameter(op.getPhi());
    if (!theta || !phi) {
      return failure();
    }

    const auto thetaValue = theta.getValueAsDouble();
    const auto phiValue = phi.getValueAsDouble();
    if (std::abs(thetaValue) > TOLERANCE || std::abs(phiValue) > TOLERANCE) {
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
    const auto phi = UOp::getStaticParameter(op.getPhi());
    const auto lambda = UOp::getStaticParameter(op.getLambda());
    if (!phi || !lambda) {
      return failure();
    }

    const auto phiValue = phi.getValueAsDouble();
    const auto lambdaValue = lambda.getValueAsDouble();
    if (std::abs(phiValue + std::numbers::pi / 2.0) > TOLERANCE ||
        std::abs(lambdaValue - std::numbers::pi / 2.0) > TOLERANCE) {
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
    const auto phi = UOp::getStaticParameter(op.getPhi());
    const auto lambda = UOp::getStaticParameter(op.getLambda());
    if (!phi || !lambda) {
      return failure();
    }

    const auto phiValue = phi.getValueAsDouble();
    const auto lambdaValue = lambda.getValueAsDouble();
    if (std::abs(phiValue) > TOLERANCE || std::abs(lambdaValue) > TOLERANCE) {
      return failure();
    }

    auto ryOp =
        rewriter.create<RYOp>(op.getLoc(), op.getInputQubit(0), op.getTheta());
    rewriter.replaceOp(op, ryOp.getResult());

    return success();
  }
};

} // namespace

void UOp::build(OpBuilder& builder, OperationState& state, const Value qubitIn,
                const std::variant<double, Value>& theta,
                const std::variant<double, Value>& phi,
                const std::variant<double, Value>& lambda) {
  const auto& thetaOperand = variantToValue(builder, state, theta);
  const auto& phiOperand = variantToValue(builder, state, phi);
  const auto& lambdaOperand = variantToValue(builder, state, lambda);
  build(builder, state, qubitIn, thetaOperand, phiOperand, lambdaOperand);
}

void UOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<ReplaceUWithP, ReplaceUWithRX, ReplaceUWithRY>(context);
}
