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
 * @brief Merge subsequent RY operations on the same qubit by adding their
 * angles.
 */
struct MergeSubsequentRY final : OpRewritePattern<RYOp> {
  using OpRewritePattern::OpRewritePattern;

  /**
   * @brief Matches consecutive `RY` operations on the same qubit and merges them into a single `RY` with the combined rotation angle.
   *
   * @param op The `RY` operation to examine for a merge opportunity.
   * @param rewriter Pattern rewriter used to perform the transformation when a match is found.
   * @return LogicalResult `success` if the pattern was applied and the rewrite performed, `failure` otherwise.
   */
  LogicalResult matchAndRewrite(RYOp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetOneParameter(op, rewriter);
  }
};

} /**
 * @brief Constructs an RY operation from a qubit and a theta specified as either a literal or an SSA value.
 *
 * @param qubitIn Qubit value that the rotation will target.
 * @param theta Angle for the rotation, provided either as a `double` (radians) or as an MLIR `Value` that yields the angle.
 */

void RYOp::build(OpBuilder& odsBuilder, OperationState& odsState, Value qubitIn,
                 const std::variant<double, Value>& theta) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, qubitIn, thetaOperand);
}

/**
 * @brief Folds an RY rotation into its input when the rotation angle is statically zero within tolerance.
 *
 * If the `theta` operand is a compile-time constant and its absolute value is <= TOLERANCE, the operation is folded away.
 *
 * @return OpFoldResult The input qubit `Value` to replace this operation when folded, or an empty `OpFoldResult` if no folding applies.
 */
OpFoldResult RYOp::fold(FoldAdaptor /*adaptor*/) {
  if (const auto theta = valueToDouble(getTheta());
      theta && std::abs(*theta) <= TOLERANCE) {
    return getInputQubit(0);
  }
  return {};
}

/**
 * @brief Register canonicalization patterns for RY operations.
 *
 * Adds the pattern that merges consecutive RY operations acting on the same
 * qubit to the provided rewrite pattern set.
 *
 * @param results Pattern set to populate with RY canonicalization patterns.
 * @param context MLIR context used to construct the rewrite patterns.
 */
void RYOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<MergeSubsequentRY>(context);
}

/**
 * @brief Computes the 2x2 unitary matrix for this RY rotation when its angle is constant.
 *
 * When the rotation angle `theta` is statically known, returns the Y-rotation unitary
 * matrix:
 *   [[cos(theta/2), -sin(theta/2)],
 *    [ sin(theta/2),  cos(theta/2)]]
 *
 * @return std::optional<Eigen::Matrix2cd> The 2x2 unitary matrix for the RY rotation if `theta`
 *         is available as a constant; `std::nullopt` otherwise.
 */
std::optional<Eigen::Matrix2cd> RYOp::getUnitaryMatrix() {
  if (const auto theta = valueToDouble(getTheta())) {
    const auto m00 = std::complex<double>{std::cos(*theta / 2.0)};
    const auto m01 = std::complex<double>{-std::sin(*theta / 2.0)};
    return Eigen::Matrix2cd{{m00, m01}, {-m01, m00}};
  }
  return std::nullopt;
}
