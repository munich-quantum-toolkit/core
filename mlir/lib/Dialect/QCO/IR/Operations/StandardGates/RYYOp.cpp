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
#include <mlir/Support/LLVM.h>
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
 * @brief Merge subsequent RYY operations on the same qubits by adding their
 * angles.
 */
struct MergeSubsequentRYY final : OpRewritePattern<RYYOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RYYOp op,
                                PatternRewriter& rewriter) const override {
    return mergeTwoTargetOneParameter(op, rewriter);
  }
};

/**
 * @brief Merge subsequent RYY operations with swapped targets by adding their
 * angles.
 */
struct MergeSwappedTargetsRYY final : OpRewritePattern<RYYOp> {
  using OpRewritePattern::OpRewritePattern;

  /**
   * @brief Matches an RYY operation whose neighboring RYY can be merged by
   * swapping targets, and performs the merged rewrite.
   *
   * @param op The RYY operation to match.
   * @param rewriter The rewriter used to apply the transformation.
   * @return LogicalResult `success()` if the operation was merged and
   * rewritten, `failure()` otherwise.
   */
  LogicalResult matchAndRewrite(RYYOp op,
                                PatternRewriter& rewriter) const override {
    return mergeTwoTargetOneParameterWithSwappedTargets(op, rewriter);
  }
};

} // namespace

void RYYOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  Value qubit0In, Value qubit1In,
                  const std::variant<double, Value>& theta) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand);
}

/**
 * @brief Canonicalizes the op by folding it away when the rotation angle is
 * effectively zero.
 *
 * If the `theta` operand can be converted to a `double` and its absolute value
 * is less than or equal to `TOLERANCE`, the operation is replaced by its two
 * input qubit values.
 *
 * @param results Container to which the op's folded results are appended; when
 * folding occurs the two input qubit Values are emplaced into this vector.
 * @return LogicalResult `success()` if the op was folded (theta ≈ 0 and results
 * populated), `failure()` otherwise.
 */
LogicalResult RYYOp::fold(FoldAdaptor /*adaptor*/,
                          SmallVectorImpl<OpFoldResult>& results) {
  if (const auto theta = valueToDouble(getTheta());
      theta && std::abs(*theta) <= TOLERANCE) {
    results.emplace_back(getInputQubit(0));
    results.emplace_back(getInputQubit(1));
    return success();
  }
  return failure();
}

/**
 * @brief Register canonicalization rewrite patterns for RYYOp.
 *
 * Adds the MergeSubsequentRYY and MergeSwappedTargetsRYY rewrite patterns to
 * the provided pattern list so they are available for canonicalization.
 *
 * @param results Pattern list to populate with RYYOp canonicalization patterns.
 * @param context MLIR context used to construct the patterns.
 */
void RYYOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<MergeSubsequentRYY, MergeSwappedTargetsRYY>(context);
}

/**
 * @brief Compute the 4x4 unitary matrix representing the RYY rotation in the
 * computational basis.
 *
 * The returned matrix corresponds to the two-qubit RYY(theta) gate for the
 * gate's current `theta` value.
 *
 * @return std::optional<Eigen::Matrix4cd> A 4x4 complex matrix for RYY(theta)
 * if `theta` can be converted to a `double`; `std::nullopt` if `theta` is not
 * statically convertible to a numeric value.
 */
std::optional<Eigen::Matrix4cd> RYYOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  if (const auto theta = valueToDouble(getTheta())) {
    const auto m0 = 0i;
    const auto mc = std::complex<double>{std::cos(*theta / 2.0)};
    const auto ms = std::complex<double>{0.0, std::sin(*theta / 2.0)};
    return Eigen::Matrix4cd{{mc, m0, m0, ms},  // row 0
                            {m0, mc, -ms, m0}, // row 1
                            {m0, -ms, mc, m0}, // row 2
                            {ms, m0, m0, mc}}; // row 3
  }
  return std::nullopt;
}
