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
 * @brief Merge subsequent RZ operations on the same qubit by adding their
 * angles.
 */
struct MergeSubsequentRZ final : OpRewritePattern<RZOp> {
  using OpRewritePattern::OpRewritePattern;

  /**
   * @brief Attempts to merge this RZ operation with a subsequent RZ on the same qubit.
   *
   * @param op The RZ operation to match and potentially rewrite.
   * @param rewriter Rewriter used to apply the merge when applicable.
   * @return LogicalResult `success` if a merge was applied, `failure` otherwise.
   */
  LogicalResult matchAndRewrite(RZOp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetOneParameter(op, rewriter);
  }
};

} /**
 * @brief Construct an RZ operation from a qubit and a theta that may be either a constant double or an MLIR Value.
 *
 * @param odsBuilder Builder used to create the operation.
 * @param odsState OperationState to populate.
 * @param qubitIn Input qubit value the RZ acts on.
 * @param theta Rotation angle as either a `double` constant or an existing MLIR `Value`; the variant form will be converted to an SSA `Value` before constructing the operation.
 */

void RZOp::build(OpBuilder& odsBuilder, OperationState& odsState, Value qubitIn,
                 const std::variant<double, Value>& theta) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, qubitIn, thetaOperand);
}

/**
 * @brief Fold trivial RZ rotations by replacing the operation with its input qubit when the angle is effectively zero.
 *
 * If `theta` is a constant and its absolute value is less than or equal to TOLERANCE, returns the operation's input qubit to indicate the RZ can be removed; otherwise indicates no folding.
 *
 * @return OpFoldResult `Value` of the input qubit when folded, empty `OpFoldResult` otherwise.
 */
OpFoldResult RZOp::fold(FoldAdaptor /*adaptor*/) {
  if (const auto theta = valueToDouble(getTheta());
      theta && std::abs(*theta) <= TOLERANCE) {
    return getInputQubit(0);
  }
  return {};
}

/**
 * @brief Register canonicalization patterns for RZOp.
 *
 * Adds the MergeSubsequentRZ rewrite pattern to the provided pattern set
 * for the given MLIR context.
 *
 * @param results Pattern set to populate with canonicalization patterns.
 * @param context MLIR context used when constructing the rewrite patterns.
 */
void RZOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<MergeSubsequentRZ>(context);
}

/**
 * @brief Compute the 2×2 unitary matrix for this RZ rotation when the rotation angle is a constant.
 *
 * When the operator's `theta` is available as a compile-time constant, returns the matrix
 * diag(e^{-i*theta/2}, e^{i*theta/2}) as an Eigen::Matrix2cd. If `theta` is not a constant,
 * indicates absence by returning `std::nullopt`.
 *
 * @return std::optional<Eigen::Matrix2cd> The 2×2 unitary matrix for the RZ rotation if `theta` is constant, `std::nullopt` otherwise.
 */
std::optional<Eigen::Matrix2cd> RZOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  if (const auto theta = valueToDouble(getTheta())) {
    const auto m00 = std::polar(1.0, -*theta / 2.0);
    const auto m01 = 0i;
    const auto m11 = std::polar(1.0, *theta / 2.0);
    return Eigen::Matrix2cd{{m00, m01}, {m01, m11}};
  }
  return std::nullopt;
}
