/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/MQT/Utils/GatePowering.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Utils/Matrix.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <cmath>
#include <numbers>
#include <optional>
#include <variant>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::mqt;

namespace {

/// Replace U(0, 0, lambda) with P(lambda).
struct ReplaceUWithP final : OpRewritePattern<UOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UOp op,
                                PatternRewriter& rewriter) const override {
    const auto theta = valueToDouble(op.getTheta());
    const auto phi = valueToDouble(op.getPhi());
    if (!theta || std::abs(*theta) > PARAMETER_COMPARISON_TOLERANCE || !phi ||
        std::abs(*phi) > PARAMETER_COMPARISON_TOLERANCE) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<POp>(op, op.getInputQubit(0), op.getLambda());
    return success();
  }
};

/// Replace U(theta, -pi / 2, pi / 2) with RX(theta).
struct ReplaceUWithRX final : OpRewritePattern<UOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UOp op,
                                PatternRewriter& rewriter) const override {
    const auto phi = valueToDouble(op.getPhi());
    const auto lambda = valueToDouble(op.getLambda());
    if (!phi ||
        std::abs(*phi + (std::numbers::pi / 2.0)) >
            PARAMETER_COMPARISON_TOLERANCE ||
        !lambda ||
        std::abs(*lambda - (std::numbers::pi / 2.0)) >
            PARAMETER_COMPARISON_TOLERANCE) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<RXOp>(op, op.getInputQubit(0), op.getTheta());
    return success();
  }
};

/// Replace U(theta, 0, 0) with RY(theta).
struct ReplaceUWithRY final : OpRewritePattern<UOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UOp op,
                                PatternRewriter& rewriter) const override {
    const auto phi = valueToDouble(op.getPhi());
    const auto lambda = valueToDouble(op.getLambda());
    if (!phi || std::abs(*phi) > PARAMETER_COMPARISON_TOLERANCE || !lambda ||
        std::abs(*lambda) > PARAMETER_COMPARISON_TOLERANCE) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<RYOp>(op, op.getInputQubit(0), op.getTheta());
    return success();
  }
};

/// Replace U(pi / 2, phi, lambda) with U2(phi, lambda).
struct ReplaceUWithU2 final : OpRewritePattern<UOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UOp op,
                                PatternRewriter& rewriter) const override {
    const auto theta = valueToDouble(op.getTheta());
    if (!theta || std::abs(*theta - (std::numbers::pi / 2.0)) >
                      PARAMETER_COMPARISON_TOLERANCE) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<U2Op>(op, op.getInputQubit(0), op.getPhi(),
                                      op.getLambda());
    return success();
  }
};

} // namespace

void UOp::build(OpBuilder& odsBuilder, OperationState& odsState, Value qubitIn,
                const std::variant<double, Value>& theta,
                const std::variant<double, Value>& phi,
                const std::variant<double, Value>& lambda) {
  auto thetaOperand = variantToValue(odsBuilder, odsState.location, theta);
  auto phiOperand = variantToValue(odsBuilder, odsState.location, phi);
  auto lambdaOperand = variantToValue(odsBuilder, odsState.location, lambda);
  build(odsBuilder, odsState, qubitIn, thetaOperand, phiOperand, lambdaOperand);
}

void UOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<ReplaceUWithP, ReplaceUWithRX, ReplaceUWithRY, ReplaceUWithU2>(
      context);
}

Matrix2x2 UOp::unitaryMatrix(double theta, double phi, double lambda) {
  return {.data = computeUMatrix(theta, phi, lambda)};
}

std::optional<Matrix2x2> UOp::getUnitaryMatrix() {
  const auto theta = valueToDouble(getTheta());
  const auto phi = valueToDouble(getPhi());
  const auto lambda = valueToDouble(getLambda());
  if (!theta || !phi || !lambda) {
    return std::nullopt;
  }
  return unitaryMatrix(*theta, *phi, *lambda);
}
