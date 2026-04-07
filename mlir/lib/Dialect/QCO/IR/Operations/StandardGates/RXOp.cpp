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

#include <Eigen/Core>
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
 * @brief Merge subsequent RX operations on the same qubit by adding their
 * angles.
 */
struct MergeSubsequentRX final : OpRewritePattern<RXOp> {
  using OpRewritePattern::OpRewritePattern;

  /**
   * @brief Combines consecutive RX operations on the same target into a single
   * RX with an accumulated rotation angle.
   *
   * @param op The RX operation to match and attempt to merge with adjacent RX
   * operations targeting the same qubit.
   * @param rewriter The PatternRewriter used to perform the rewrite if a merge
   * is applied.
   * @return LogicalResult `success` if a rewrite (merge) was performed,
   * `failure` otherwise.
   */
  LogicalResult matchAndRewrite(RXOp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetOneParameter(op, rewriter);
  }
};

} // namespace

void RXOp::build(OpBuilder& odsBuilder, OperationState& odsState, Value qubitIn,
                 const std::variant<double, Value>& theta) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, qubitIn, thetaOperand);
}

/**
 * @brief Fold an `RX` operation when its rotation angle is effectively zero.
 *
 * If the operation's `theta` operand can be converted to a numeric value and
 * its absolute value is less than or equal to `TOLERANCE`, fold the `RX` by
 * returning the operation's input qubit (the rotation is a no-op).
 *
 * @return OpFoldResult The input qubit `Value` when `theta` is within
 * `TOLERANCE`, otherwise an empty `OpFoldResult`.
 */
OpFoldResult RXOp::fold(FoldAdaptor /*adaptor*/) {
  if (const auto theta = valueToDouble(getTheta());
      theta && std::abs(*theta) <= TOLERANCE) {
    return getInputQubit(0);
  }
  return {};
}

/**
 * @brief Register canonicalization patterns for RXOp.
 *
 * Adds the pattern that merges consecutive RX operations targeting the same
 * qubit by accumulating their rotation parameter.
 *
 * @param results Container to which canonicalization patterns are added.
 * @param context MLIR context used to construct the pattern.
 */
void RXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<MergeSubsequentRX>(context);
}

/**
 * @brief Computes the 2x2 unitary matrix for an RX gate using the operation's
 * theta.
 *
 * Produces the SU(2) rotation matrix for a rotation about the X axis by angle
 * theta.
 *
 * @return std::optional<Eigen::Matrix2cd> The matrix
 * [[cos(theta/2), -i*sin(theta/2)], [-i*sin(theta/2), cos(theta/2)]] if theta
 * can be converted to a double; `std::nullopt` if theta is not a concrete
 * numeric value.
 */
std::optional<Eigen::Matrix2cd> RXOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  if (const auto theta = valueToDouble(getTheta())) {
    const auto m00 = std::cos(*theta / 2.0) + 0i;
    const auto m01 = -1i * std::sin(*theta / 2.0);
    return Eigen::Matrix2cd{{m00, m01}, {m01, m00}};
  }
  return std::nullopt;
}
