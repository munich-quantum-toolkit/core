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
 * @brief Merge subsequent RXX operations on the same qubits by adding their
 * angles.
 */
struct MergeSubsequentRXX final : OpRewritePattern<RXXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RXXOp op,
                                PatternRewriter& rewriter) const override {
    return mergeTwoTargetOneParameter(op, rewriter);
  }
};

/**
 * @brief Merge subsequent RXX operations with swapped targets by adding their
 * angles.
 */
struct MergeSwappedTargetsRXX final : OpRewritePattern<RXXOp> {
  using OpRewritePattern::OpRewritePattern;

  /**
   * @brief Attempts to merge this RXX operation with a subsequent RXX that has swapped targets.
   *
   * Tries to combine two consecutive RXX operations acting on the same qubits with their targets swapped
   * by merging their rotation parameters and performing the corresponding rewrite.
   *
   * @param op The RXX operation to examine for merging.
   * @param rewriter The pattern rewriter used to apply the merge transformation.
   * @return LogicalResult `success()` if a merge and rewrite were performed, `failure()` otherwise.
   */
  LogicalResult matchAndRewrite(RXXOp op,
                                PatternRewriter& rewriter) const override {
    return mergeTwoTargetOneParameterWithSwappedTargets(op, rewriter);
  }
};

} /**
 * @brief Builds an RXX operation accepting `theta` as either a plain double or an SSA `Value`.
 *
 * Constructs the operation using `qubit0In` and `qubit1In` as the two target qubits and
 * accepts `theta` either as a numeric angle in radians or as an MLIR `Value` that yields the angle.
 *
 * @param odsBuilder Builder used to create IR (implicit construction context).
 * @param odsState OperationState to populate (implicit construction target).
 * @param qubit0In First target qubit value.
 * @param qubit1In Second target qubit value.
 * @param theta Angle parameter either as `double` (radians) or as an MLIR `Value` producing the angle.
 */

void RXXOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  Value qubit0In, Value qubit1In,
                  const std::variant<double, Value>& theta) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand);
}

/**
 * @brief Folds the RXX operation into its input qubits when the rotation angle is effectively zero.
 *
 * If the `theta` operand is a constant whose absolute value is less than or equal to the internal
 * tolerance, this method appends the two input qubit values to `results` as replacement values
 * and indicates the operation was folded.
 *
 * @param results Container to receive replacement OpFoldResult values; on success two input
 *                qubits (indices 0 and 1) are appended.
 * @return LogicalResult `success()` if the operation was folded (and replacements were produced),
 *         `failure()` otherwise.
 */
LogicalResult RXXOp::fold(FoldAdaptor /*adaptor*/,
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
 * @brief Populate the given pattern set with RXX-specific canonicalization patterns.
 *
 * Adds the rewrite patterns that merge consecutive RXX operations (including cases
 * with swapped target qubits) into the provided pattern set.
 *
 * @param results Pattern set to populate with canonicalization patterns.
 * @param context MLIR context used to construct the patterns.
 */
void RXXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<MergeSubsequentRXX, MergeSwappedTargetsRXX>(context);
}

/**
 * @brief Compute the two-qubit RXX gate's 4x4 unitary matrix for the current `theta`.
 *
 * If `theta` can be resolved to a concrete double, returns the 4x4 complex matrix
 * implementing exp(-i * theta/2 * X ⊗ X) with entries built from cos(theta/2) and -i*sin(theta/2).
 *
 * @return std::optional<Eigen::Matrix4cd> The 4x4 complex unitary when `theta` is available, `std::nullopt` otherwise.
 */
std::optional<Eigen::Matrix4cd> RXXOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  if (const auto theta = valueToDouble(getTheta())) {
    const auto m0 = 0i;
    const auto mc = std::cos(*theta / 2.0) + 0i;
    const auto ms = -1i * std::sin(*theta / 2.0);
    return Eigen::Matrix4cd{{mc, m0, m0, ms},  // row 0
                            {m0, mc, ms, m0},  // row 1
                            {m0, ms, mc, m0},  // row 2
                            {ms, m0, m0, mc}}; // row 3
  }
  return std::nullopt;
}
