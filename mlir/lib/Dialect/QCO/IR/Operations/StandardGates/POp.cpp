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
 * @brief Merge subsequent P operations on the same qubit by adding their
 * angles.
 */
struct MergeSubsequentP final : OpRewritePattern<POp> {
  using OpRewritePattern::OpRewritePattern;

  /**
   * @brief Attempts to merge consecutive single-target, single-parameter `P`
   * operations on the same qubit into a single `P`.
   *
   * @returns `success` if the pattern was applied and the operation was
   * rewritten, `failure` otherwise.
   */
  LogicalResult matchAndRewrite(POp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetOneParameter(op, rewriter);
  }
};

} // namespace

void POp::build(OpBuilder& odsBuilder, OperationState& odsState, Value qubitIn,
                const std::variant<double, Value>& theta) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, qubitIn, thetaOperand);
}

/**
 * @brief Fold the P gate into its input when the rotation angle is effectively
 * zero.
 *
 * If `theta` is statically known and `abs(theta) <= TOLERANCE`, the operation
 * can be folded away and replaced by its single input qubit.
 *
 * @returns The input qubit `Value` to use as the folding result when `theta` is
 * within tolerance; an empty `OpFoldResult` otherwise.
 */
OpFoldResult POp::fold(FoldAdaptor /*adaptor*/) {
  if (const auto theta = valueToDouble(getTheta());
      theta && std::abs(*theta) <= TOLERANCE) {
    return getInputQubit(0);
  }
  return {};
}

/**
 * @brief Register canonicalization patterns for POp.
 *
 * Adds the MergeSubsequentP pattern to the provided rewrite pattern set so
 * adjacent P operations on the same qubit can be merged during
 * canonicalization.
 *
 * @param results Pattern set to populate with canonicalization patterns.
 * @param context MLIR context used to construct the patterns.
 */
void POp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<MergeSubsequentP>(context);
}

/**
 * @brief Compute the 2×2 unitary matrix represented by this P gate when its
 * rotation angle is constant.
 *
 * If the gate's theta is a compile-time constant, returns the diagonal matrix
 * diag(1, e^{i theta}).
 *
 * @return std::optional<Eigen::Matrix2cd> The 2×2 complex unitary matrix when
 * theta is known, `std::nullopt` when theta is not a constant.
 */
std::optional<Eigen::Matrix2cd> POp::getUnitaryMatrix() {
  if (const auto theta = valueToDouble(getTheta())) {
    return Eigen::Matrix2cd{{1.0, 0.0}, {0.0, std::polar(1.0, *theta)}};
  }
  return std::nullopt;
}
