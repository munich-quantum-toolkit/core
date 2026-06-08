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
#include "mlir/Dialect/Utils/Utils.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <complex>
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

} // namespace

void RZOp::build(OpBuilder& odsBuilder, OperationState& odsState, Value qubitIn,
                 const std::variant<double, Value>& theta) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, qubitIn, thetaOperand);
}

OpFoldResult RZOp::fold(FoldAdaptor /*adaptor*/) {
  if (const auto theta = valueToDouble(getTheta());
      theta && std::abs(*theta) <= TOLERANCE) {
    return getInputQubit(0);
  }
  return {};
}

void RZOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<MergeSubsequentRZ>(context);
}

std::optional<Matrix2> RZOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  if (const auto theta = valueToDouble(getTheta())) {
    const auto m00 = std::polar(1.0, -*theta / 2.0);
    const auto m01 = 0i;
    const auto m10 = m01;
    const auto m11 = std::polar(1.0, *theta / 2.0);
    return Matrix2::fromElements(m00, m01, m10, m11);
  }
  return std::nullopt;
}
