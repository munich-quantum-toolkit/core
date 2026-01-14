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
 * @brief Replace U2(0, pi) with H.
 */
struct ReplaceU2WithH final : OpRewritePattern<U2Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(U2Op op,
                                PatternRewriter& rewriter) const override {
    const auto phi = valueToDouble(op.getPhi());
    const auto lambda = valueToDouble(op.getLambda());
    if (!phi || std::abs(*phi) > TOLERANCE || !lambda ||
        std::abs(*lambda - std::numbers::pi) > TOLERANCE) {
      return failure();
    }

    auto hOp = rewriter.create<HOp>(op.getLoc(), op.getInputQubit(0));
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
    const auto phi = valueToDouble(op.getPhi());
    const auto lambda = valueToDouble(op.getLambda());
    if (!phi || std::abs(*phi + (std::numbers::pi / 2.0)) > TOLERANCE ||
        !lambda || std::abs(*lambda - (std::numbers::pi / 2.0)) > TOLERANCE) {
      return failure();
    }

    auto rxOp = rewriter.create<RXOp>(op.getLoc(), op.getInputQubit(0),
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
    const auto phi = valueToDouble(op.getPhi());
    const auto lambda = valueToDouble(op.getLambda());
    if (!phi || std::abs(*phi) > TOLERANCE || !lambda ||
        std::abs(*lambda) > TOLERANCE) {
      return failure();
    }

    auto ryOp = rewriter.create<RYOp>(op.getLoc(), op.getInputQubit(0),
                                      std::numbers::pi / 2.0);
    rewriter.replaceOp(op, ryOp.getResult());

    return success();
  }
};

} // namespace

void U2Op::build(OpBuilder& builder, OperationState& state, Value qubitIn,
                 const std::variant<double, Value>& phi,
                 const std::variant<double, Value>& lambda) {
  auto phiOperand = variantToValue(builder, state.location, phi);
  auto lambdaOperand = variantToValue(builder, state.location, lambda);
  build(builder, state, qubitIn, phiOperand, lambdaOperand);
}

void U2Op::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<ReplaceU2WithH, ReplaceU2WithRX, ReplaceU2WithRY>(context);
}

std::optional<Eigen::Matrix2cd> U2Op::getUnitaryMatrix() {
  using namespace std::complex_literals;

  if (auto phi = utils::valueToDouble(getPhi())) {
    if (auto lambda = utils::valueToDouble(getLambda())) {
      const auto m00 = 1.0 / std::numbers::sqrt2 + 0i;
      const auto m01 =
          std::polar(1.0 / std::numbers::sqrt2, *lambda + std::numbers::pi);
      const auto m10 = std::polar(1.0 / std::numbers::sqrt2, *phi);
      const auto m11 = std::polar(1.0 / std::numbers::sqrt2, *phi + *lambda);
      return Eigen::Matrix2cd{{m00, m01}, {m10, m11}};
    }
  }
  return std::nullopt;
}
