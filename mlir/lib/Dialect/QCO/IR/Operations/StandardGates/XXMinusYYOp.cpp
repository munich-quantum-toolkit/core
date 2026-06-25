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
 * @brief Merge subsequent XXMinusYY operations on the same qubits by adding
 * their thetas.
 */
struct MergeSubsequentXXMinusYY final : OpRewritePattern<XXMinusYYOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(XXMinusYYOp op,
                                PatternRewriter& rewriter) const override {
    return mergeXXPlusMinusYY(op, rewriter);
  }
};

} // namespace

void XXMinusYYOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                        Value qubit0In, Value qubit1In,
                        const std::variant<double, Value>& theta,
                        const std::variant<double, Value>& beta) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  const auto betaOperand = variantToValue(odsBuilder, odsState.location, beta);
  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand, betaOperand);
}

LogicalResult XXMinusYYOp::fold(FoldAdaptor /*adaptor*/,
                                SmallVectorImpl<OpFoldResult>& results) {
  if (const auto theta = valueToDouble(getTheta());
      theta && std::abs(*theta) <= TOLERANCE) {
    results.emplace_back(getInputQubit(0));
    results.emplace_back(getInputQubit(1));
    return success();
  }
  return failure();
}

void XXMinusYYOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<MergeSubsequentXXMinusYY>(context);
}

std::optional<Matrix4x4> XXMinusYYOp::getUnitaryMatrix() {
  const auto theta = valueToDouble(getTheta());
  const auto beta = valueToDouble(getBeta());
  if (!theta || !beta) {
    return std::nullopt;
  }

  using namespace std::complex_literals;
  const auto mc = std::cos(*theta / 2);
  const auto s = std::sin(*theta / 2);
  const auto msp = s * std::exp(1i * (*beta - (std::numbers::pi / 2)));
  const auto msm = s * std::exp(1i * (-*beta - (std::numbers::pi / 2)));
  return Matrix4x4::fromElements(mc, 0, 0, msm,  // row 0
                                 0, 1, 0, 0,     // row 1
                                 0, 0, 1, 0,     // row 2
                                 msp, 0, 0, mc); // row 3
}
