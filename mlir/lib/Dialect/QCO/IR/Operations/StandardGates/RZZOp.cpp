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
 * @brief Merge subsequent RZZ operations on the same qubits by adding their
 * angles.
 */
struct MergeSubsequentRZZ final : OpRewritePattern<RZZOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RZZOp op,
                                PatternRewriter& rewriter) const override {
    return mergeTwoTargetOneParameter(op, rewriter);
  }
};

/**
 * @brief Merge subsequent RZZ operations with swapped targets by adding their
 * angles.
 */
struct MergeSwappedTargetsRZZ final : OpRewritePattern<RZZOp> {
  using OpRewritePattern::OpRewritePattern;

  /**
   * @brief Try to canonicalize by merging this `RZZOp` with a subsequent
   * `RZZOp` that has the same parameter but swapped target qubits.
   *
   * Attempts to combine the two operations into a single `RZZOp` (adjusting the
   * parameter as needed) by delegating to the merge helper.
   *
   * @param op The `RZZOp` to match and rewrite.
   * @param rewriter Pattern rewriter used to perform the transformation.
   * @return LogicalResult `success()` if the two `RZZOp` instances were merged,
   * `failure()` otherwise.
   */
  LogicalResult matchAndRewrite(RZZOp op,
                                PatternRewriter& rewriter) const override {
    return mergeTwoTargetOneParameterWithSwappedTargets(op, rewriter);
  }
};

} // namespace

void RZZOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  Value qubit0In, Value qubit1In,
                  const std::variant<double, Value>& theta) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand);
}

/**
 * Fold the RZZ operation when its rotation angle is effectively zero.
 *
 * If the operation's `theta` is a compile-time constant whose absolute value
 * is less than or equal to TOLERANCE, the op is folded by returning its two
 * input qubit operands as replacement values.
 *
 * @param results Container to which replacement operands are appended on
 *                successful folding; the two input qubits are pushed when
 *                folding occurs.
 * @return LogicalResult `success()` if `theta` is a known constant with
 *         absolute value <= TOLERANCE (and the two input qubits were appended
 *         to `results`), `failure()` otherwise.
 */
LogicalResult RZZOp::fold(FoldAdaptor /*adaptor*/,
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
 * @brief Registers canonicalization patterns for RZZOp.
 *
 * Adds rewrite patterns that merge consecutive RZZ operations acting on the
 * same qubits and those with swapped target qubits.
 *
 * @param results Pattern list to which the canonicalization patterns are added.
 * @param context MLIR context used to construct the patterns.
 */
void RZZOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<MergeSubsequentRZZ, MergeSwappedTargetsRZZ>(context);
}

/**
 * @brief Compute the 4×4 unitary matrix implemented by this RZZ operation when
 * the rotation angle is available.
 *
 * The matrix is diagonal with entries [e^{-i theta/2}, e^{i theta/2}, e^{i
 * theta/2}, e^{-i theta/2}].
 *
 * @return std::optional<Eigen::Matrix4cd> The 4×4 complex unitary matrix when
 * `theta` can be resolved to a double; `std::nullopt` if `theta` is not
 * available.
 */
std::optional<Eigen::Matrix4cd> RZZOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  if (const auto theta = valueToDouble(getTheta())) {
    const auto m0 = 0i;
    const auto mp = std::polar(1.0, *theta / 2.0);
    const auto mm = std::polar(1.0, -*theta / 2.0);
    return Eigen::Matrix4cd{{mm, m0, m0, m0},  // row 0
                            {m0, mp, m0, m0},  // row 1
                            {m0, m0, mp, m0},  // row 2
                            {m0, m0, m0, mm}}; // row 3
  }
  return std::nullopt;
}
