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
    rewriter.replaceOpWithNewOp<POp>(op, op.getInputQubit(0), op.getLambda());
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
    rewriter.replaceOpWithNewOp<RXOp>(op, op.getInputQubit(0), op.getTheta());
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
    rewriter.replaceOpWithNewOp<RYOp>(op, op.getInputQubit(0), op.getTheta());
    return success();
  }
};

} // namespace

void UOp::build(OpBuilder& odsBuilder, OperationState& odsState, Value qubitIn,
                const std::variant<double, Value>& theta,
                const std::variant<double, Value>& phi,
                const std::variant<double, Value>& lambda) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  const auto phiOperand = variantToValue(odsBuilder, odsState.location, phi);
  const auto lambdaOperand =
      variantToValue(odsBuilder, odsState.location, lambda);
  build(odsBuilder, odsState, qubitIn, thetaOperand, phiOperand, lambdaOperand);
}

void UOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<ReplaceUWithP, ReplaceUWithRX, ReplaceUWithRY>(context);
}

std::optional<Eigen::Matrix2cd> UOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  const auto theta = valueToDouble(getTheta());
  const auto phi = valueToDouble(getPhi());
  const auto lambda = valueToDouble(getLambda());
  if (!theta || !phi || !lambda) {
    return std::nullopt;
  }

  const auto c = std::cos(*theta / 2.0);
  const auto s = std::sin(*theta / 2.0);
  const auto m00 = c + 0i;
  const auto m01 = std::polar(s, *lambda + std::numbers::pi);
  const auto m10 = std::polar(s, *phi);
  const auto m11 = std::polar(c, *phi + *lambda);
  return Eigen::Matrix2cd{{m00, m01}, {m10, m11}};
}
