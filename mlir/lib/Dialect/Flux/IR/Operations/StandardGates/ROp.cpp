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

#include <cmath>
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
 * @brief Replace R(theta, 0) with RX(theta).
 */
struct ReplaceRWithRX final : OpRewritePattern<ROp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ROp op,
                                PatternRewriter& rewriter) const override {
    const auto phi = ROp::getStaticParameter(op.getPhi());
    if (!phi) {
      return failure();
    }

    const auto phiValue = phi.getValueAsDouble();
    if (std::abs(phiValue) > TOLERANCE) {
      return failure();
    }

    auto rxOp =
        rewriter.create<RXOp>(op.getLoc(), op.getInputQubit(0), op.getTheta());
    rewriter.replaceOp(op, rxOp.getResult());

    return success();
  }
};

/**
 * @brief Replace R(theta, pi / 2) with RY(theta).
 */
struct ReplaceRWithRY final : OpRewritePattern<ROp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ROp op,
                                PatternRewriter& rewriter) const override {
    const auto phi = ROp::getStaticParameter(op.getPhi());
    if (!phi) {
      return failure();
    }

    const auto phiValue = phi.getValueAsDouble();
    if (std::abs(phiValue - (std::numbers::pi / 2.0)) > TOLERANCE) {
      return failure();
    }

    auto ryOp =
        rewriter.create<RYOp>(op.getLoc(), op.getInputQubit(0), op.getTheta());
    rewriter.replaceOp(op, ryOp.getResult());

    return success();
  }
};

} // namespace

void ROp::build(OpBuilder& builder, OperationState& state, Value qubitIn,
                const std::variant<double, Value>& theta,
                const std::variant<double, Value>& phi) {
  const auto& thetaOperand = variantToValue(builder, state, theta);
  const auto& phiOperand = variantToValue(builder, state, phi);
  build(builder, state, qubitIn, thetaOperand, phiOperand);
}

void ROp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<ReplaceRWithRX, ReplaceRWithRY>(context);
}
