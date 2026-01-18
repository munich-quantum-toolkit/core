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
 * @brief Replace R(theta, 0) with RX(theta).
 */
struct ReplaceRWithRX final : OpRewritePattern<ROp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ROp op,
                                PatternRewriter& rewriter) const override {
    const auto phi = valueToDouble(op.getPhi());
    if (!phi || std::abs(*phi) > TOLERANCE) {
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
    const auto phi = valueToDouble(op.getPhi());
    if (!phi || std::abs(*phi - (std::numbers::pi / 2.0)) > TOLERANCE) {
      return failure();
    }

    auto ryOp =
        rewriter.create<RYOp>(op.getLoc(), op.getInputQubit(0), op.getTheta());
    rewriter.replaceOp(op, ryOp.getResult());

    return success();
  }
};

} // namespace

void ROp::build(OpBuilder& odsBuilder, OperationState& odsState, Value qubitIn,
                const std::variant<double, Value>& theta,
                const std::variant<double, Value>& phi) {
  auto thetaOperand = variantToValue(odsBuilder, odsState.location, theta);
  auto phiOperand = variantToValue(odsBuilder, odsState.location, phi);
  build(odsBuilder, odsState, qubitIn, thetaOperand, phiOperand);
}

void ROp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<ReplaceRWithRX, ReplaceRWithRY>(context);
}

std::optional<Eigen::Matrix2cd> ROp::getUnitaryMatrix() {
  auto theta = valueToDouble(getTheta());
  auto phi = valueToDouble(getPhi());
  if (!theta || !phi) {
    return std::nullopt;
  }

  const auto thetaSin = std::sin(*theta / 2.0);
  const auto m01 = std::polar(thetaSin, -*phi - (std::numbers::pi / 2));
  const auto m10 = std::polar(thetaSin, *phi - (std::numbers::pi / 2));
  const std::complex<double> thetaCos = std::cos(*theta / 2.0);
  return Eigen::Matrix2cd{{thetaCos, m01}, {m10, thetaCos}};
}
