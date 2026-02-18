/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <Eigen/Core>
#include <cmath>
#include <complex>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <numbers>
#include <optional>
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
    rewriter.replaceOpWithNewOp<HOp>(op, op.getInputQubit(0));
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
    rewriter.replaceOpWithNewOp<RXOp>(op, op.getInputQubit(0),
                                      std::numbers::pi / 2.0);
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
    rewriter.replaceOpWithNewOp<RYOp>(op, op.getInputQubit(0),
                                      std::numbers::pi / 2.0);
    return success();
  }
};

} // namespace

void U2Op::build(OpBuilder& odsBuilder, OperationState& odsState, Value qubitIn,
                 const std::variant<double, Value>& phi,
                 const std::variant<double, Value>& lambda) {
  const auto phiOperand = variantToValue(odsBuilder, odsState.location, phi);
  const auto lambdaOperand =
      variantToValue(odsBuilder, odsState.location, lambda);
  build(odsBuilder, odsState, qubitIn, phiOperand, lambdaOperand);
}

void U2Op::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<ReplaceU2WithH, ReplaceU2WithRX, ReplaceU2WithRY>(context);
}

std::optional<Eigen::Matrix2cd> U2Op::getUnitaryMatrix() {
  using namespace std::complex_literals;

  const auto phi = valueToDouble(getPhi());
  const auto lambda = valueToDouble(getLambda());
  if (!phi || !lambda) {
    return std::nullopt;
  }

  const auto m00 = 1.0 / std::numbers::sqrt2 + 0i;
  const auto m01 =
      std::polar(1.0 / std::numbers::sqrt2, *lambda + std::numbers::pi);
  const auto m10 = std::polar(1.0 / std::numbers::sqrt2, *phi);
  const auto m11 = std::polar(1.0 / std::numbers::sqrt2, *phi + *lambda);
  return Eigen::Matrix2cd{{m00, m01}, {m10, m11}};
}
