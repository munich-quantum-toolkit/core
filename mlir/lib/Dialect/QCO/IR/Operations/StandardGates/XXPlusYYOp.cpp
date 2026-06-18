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

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

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
 * @brief Merge subsequent XXPlusYY operations on the same qubits by adding
 * their thetas.
 */
struct MergeSubsequentXXPlusYY final : OpRewritePattern<XXPlusYYOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(XXPlusYYOp op,
                                PatternRewriter& rewriter) const override {
    return mergeXXPlusMinusYY(op, rewriter);
  }
};

} // namespace

void XXPlusYYOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                       Value qubit0In, Value qubit1In,
                       const std::variant<double, Value>& theta,
                       const std::variant<double, Value>& beta) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  const auto betaOperand = variantToValue(odsBuilder, odsState.location, beta);
  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand, betaOperand);
}

LogicalResult XXPlusYYOp::fold(FoldAdaptor /*adaptor*/,
                               SmallVectorImpl<OpFoldResult>& results) {
  if (const auto theta = valueToDouble(getTheta());
      theta && std::abs(*theta) <= TOLERANCE) {
    results.emplace_back(getInputQubit(0));
    results.emplace_back(getInputQubit(1));
    return success();
  }
  return failure();
}

void XXPlusYYOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.add<MergeSubsequentXXPlusYY>(context);
}

std::optional<Matrix4x4> XXPlusYYOp::getUnitaryMatrix() {
  const auto theta = valueToDouble(getTheta());
  const auto beta = valueToDouble(getBeta());
  if (!theta || !beta) {
    return std::nullopt;
  }

  const auto mc = std::cos(*theta / 2);
  const auto s = std::sin(*theta / 2);
  const auto msp = std::polar(s, *beta - (std::numbers::pi / 2));
  const auto msm = std::polar(s, -*beta - (std::numbers::pi / 2));
  return Matrix4x4::fromElements(1, 0, 0, 0,    // row 0
                                 0, mc, msp, 0, // row 1
                                 0, msm, mc, 0, // row 2
                                 0, 0, 0, 1);   // row 3
}
