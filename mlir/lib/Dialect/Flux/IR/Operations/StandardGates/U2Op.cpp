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

namespace {

/**
 * @brief Replace U2(0, pi) with H.
 */
struct ReplaceU2WithH final : OpRewritePattern<U2Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(U2Op op,
                                PatternRewriter& rewriter) const override {
    const auto phi = U2Op::getStaticParameter(op.getPhi());
    const auto lambda = U2Op::getStaticParameter(op.getLambda());
    if (!phi || !lambda) {
      return failure();
    }

    const auto phiValue = phi.getValueAsDouble();
    const auto lambdaValue = lambda.getValueAsDouble();
    if (phiValue != 0.0 || lambdaValue != std::numbers::pi) {
      return failure();
    }

    auto hOp = rewriter.create<HOp>(op.getLoc(), op.getQubitIn());
    rewriter.replaceOp(op, hOp.getResult());

    return success();
  }
};

/**
 * @brief Replace U2(-pi / 2, pi / 2) with RX(pi / 2).
 */
struct ReplaceU2WithRX final : OpRewritePattern<U2Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(U2Op op,
                                PatternRewriter& rewriter) const override {
    const auto phi = U2Op::getStaticParameter(op.getPhi());
    const auto lambda = U2Op::getStaticParameter(op.getLambda());
    if (!phi || !lambda) {
      return failure();
    }

    const auto phiValue = phi.getValueAsDouble();
    const auto lambdaValue = lambda.getValueAsDouble();
    if (phiValue != -std::numbers::pi / 2.0 ||
        lambdaValue != std::numbers::pi / 2.0) {
      return failure();
    }

    auto rxOp = rewriter.create<RXOp>(op.getLoc(), op.getQubitIn(),
                                      std::numbers::pi / 2.0);
    rewriter.replaceOp(op, rxOp.getResult());

    return success();
  }
};

/**
 * @brief Replace U2(0, 0) with RY(pi / 2).
 */
struct ReplaceU2WithRY final : OpRewritePattern<U2Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(U2Op op,
                                PatternRewriter& rewriter) const override {
    const auto phi = U2Op::getStaticParameter(op.getPhi());
    const auto lambda = U2Op::getStaticParameter(op.getLambda());
    if (!phi || !lambda) {
      return failure();
    }

    const auto phiValue = phi.getValueAsDouble();
    const auto lambdaValue = lambda.getValueAsDouble();
    if (phiValue != 0.0 || lambdaValue != 0.0) {
      return failure();
    }

    auto ryOp = rewriter.create<RYOp>(op.getLoc(), op.getQubitIn(),
                                      std::numbers::pi / 2.0);
    rewriter.replaceOp(op, ryOp.getResult());

    return success();
  }
};

} // namespace

void U2Op::build(OpBuilder& odsBuilder, OperationState& odsState,
                 const Value qubitIn, const std::variant<double, Value>& phi,
                 const std::variant<double, Value>& lambda) {
  Value phiOperand = nullptr;
  if (std::holds_alternative<double>(phi)) {
    phiOperand = odsBuilder.create<arith::ConstantOp>(
        odsState.location, odsBuilder.getF64FloatAttr(std::get<double>(phi)));
  } else {
    phiOperand = std::get<Value>(phi);
  }

  Value lambdaOperand = nullptr;
  if (std::holds_alternative<double>(lambda)) {
    lambdaOperand = odsBuilder.create<arith::ConstantOp>(
        odsState.location,
        odsBuilder.getF64FloatAttr(std::get<double>(lambda)));
  } else {
    lambdaOperand = std::get<Value>(lambda);
  }

  build(odsBuilder, odsState, qubitIn, phiOperand, lambdaOperand);
}

void U2Op::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<ReplaceU2WithH, ReplaceU2WithRX, ReplaceU2WithRY>(context);
}
