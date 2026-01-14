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
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <Eigen/Core>
#include <complex>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>
#include <variant>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::utils;

namespace {

/**
 * @brief Merge subsequent RZ operations on the same qubit by adding their
 * angles.
 */
struct MergeSubsequentRZ final : OpRewritePattern<RZOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RZOp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetOneParameter(op, rewriter);
  }
};

/**
 * @brief Remove trivial RZ operations.
 */
struct RemoveTrivialRZ final : OpRewritePattern<RZOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RZOp op,
                                PatternRewriter& rewriter) const override {
    return removeTrivialOneTargetOneParameter(op, rewriter);
  }
};

} // namespace

void RZOp::build(OpBuilder& builder, OperationState& state, Value qubitIn,
                 const std::variant<double, Value>& theta) {
  auto thetaOperand = variantToValue(builder, state.location, theta);
  build(builder, state, qubitIn, thetaOperand);
}

void RZOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<MergeSubsequentRZ, RemoveTrivialRZ>(context);
}

std::optional<Eigen::Matrix2cd> RZOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  if (auto theta = utils::valueToDouble(getTheta())) {
    const auto m00 = std::polar(1.0, -*theta / 2.0);
    const auto m01 = 0i;
    const auto m11 = std::polar(1.0, *theta / 2.0);
    return Eigen::Matrix2cd{{m00, m01}, {m01, m11}};
  }
  return std::nullopt;
}
