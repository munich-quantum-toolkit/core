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
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <complex>
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
    if (const auto phi = valueToDouble(op.getPhi());
        !phi || std::abs(*phi) > TOLERANCE) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<RXOp>(op, op.getInputQubit(0), op.getTheta());
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
    if (const auto phi = valueToDouble(op.getPhi());
        !phi || std::abs(*phi - (std::numbers::pi / 2.0)) > TOLERANCE) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<RYOp>(op, op.getInputQubit(0), op.getTheta());
    return success();
  }
};

/**
 * @brief Merge subsequent R operations on the same qubit with matching `phi`.
 */
struct MergeSubsequentR final : OpRewritePattern<ROp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ROp op,
                                PatternRewriter& rewriter) const override {
    auto nextOp = dyn_cast<ROp>(*op.getOutputQubit(0).user_begin());
    if (!nextOp) {
      return failure();
    }

    if (!valuesMatchWithinTolerance(op.getPhi(), nextOp.getPhi())) {
      return failure();
    }

    auto newParameter = arith::AddFOp::create(rewriter, op.getLoc(),
                                              op.getTheta(), nextOp.getTheta());
    op->setOperand(1, newParameter.getResult());
    rewriter.replaceOp(nextOp, op.getResult());
    return success();
  }
};

} // namespace

void ROp::build(OpBuilder& odsBuilder, OperationState& odsState, Value qubitIn,
                const std::variant<double, Value>& theta,
                const std::variant<double, Value>& phi) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  const auto phiOperand = variantToValue(odsBuilder, odsState.location, phi);
  build(odsBuilder, odsState, qubitIn, thetaOperand, phiOperand);
}

void ROp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<ReplaceRWithRX, ReplaceRWithRY, MergeSubsequentR>(context);
}

std::optional<Matrix2x2> ROp::getUnitaryMatrix() {
  const auto theta = valueToDouble(getTheta());
  const auto phi = valueToDouble(getPhi());
  if (!theta || !phi) {
    return std::nullopt;
  }

  using namespace std::complex_literals;
  const auto halfTheta = *theta / 2;
  const auto c = std::cos(halfTheta);
  const auto s = std::sin(halfTheta);

  const auto m01 = s * std::exp(1i * (-*phi - (std::numbers::pi / 2)));
  const auto m10 = s * std::exp(1i * (*phi - (std::numbers::pi / 2)));
  return Matrix2x2::fromElements(c, m01,  // row 0
                                 m10, c); // row 1
}
