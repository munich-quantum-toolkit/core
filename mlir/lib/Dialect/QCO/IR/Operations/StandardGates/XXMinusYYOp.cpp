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
#include "mlir/Dialect/Utils/Utils.h"

#include <Eigen/Core>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

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

  /**
   * @brief Merge a following XXMinusYYOp into this one when they act on the
   * same qubits and share the same beta.
   *
   * If the immediate user of this operation's first output is another
   * XXMinusYYOp that targets the same second qubit and has an equal `beta`
   * (either numerically within TOLERANCE or identical operands), this pattern
   * adds the two `theta` parameters and updates this operation to use the
   * summed `theta`, then removes the successor by replacing it with this
   * operation's results.
   *
   * @param op The XXMinusYYOp to match and potentially merge with its immediate
   * successor.
   * @param rewriter PatternRewriter used to create the add operation for theta
   * and perform the replacement.
   * @return LogicalResult `success()` if a merge occurred, `failure()`
   * otherwise.
   */
  LogicalResult matchAndRewrite(XXMinusYYOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the successor is the same operation
    auto nextOp =
        llvm::dyn_cast<XXMinusYYOp>(*op.getOutputQubit(0).user_begin());
    if (!nextOp) {
      return failure();
    }

    // Confirm operations act on the same qubits
    if (op.getOutputQubit(1) != nextOp.getInputQubit(1)) {
      return failure();
    }

    // Confirm betas are equal
    const auto beta = valueToDouble(op.getBeta());
    const auto nextBeta = valueToDouble(nextOp.getBeta());
    if (beta && nextBeta) {
      if (std::abs(*beta - *nextBeta) > TOLERANCE) {
        return failure();
      }
    } else if (op.getBeta() != nextOp.getBeta()) {
      return failure();
    }

    // Compute and set new theta, which has index 2
    auto newParameter = arith::AddFOp::create(
        rewriter, op.getLoc(), op.getOperand(2), nextOp.getOperand(2));
    op->setOperand(2, newParameter.getResult());

    // Replace the second operation with the result of the first operation
    rewriter.replaceOp(nextOp, op.getResults());
    return success();
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

/**
 * @brief Folds the operation into its input qubits when the rotation is
 * negligible.
 *
 * If `theta` is a compile-time constant and its absolute value is less than or
 * equal to TOLERANCE, the operation is removed and its results are replaced by
 * its two input qubits.
 *
 * @param results Vector to which replacement OpFoldResults are appended on
 *                successful folding; on success this will receive the two input
 *                qubits (input qubit 0 then input qubit 1).
 * @return LogicalResult `success()` if folding occurred, `failure()` otherwise.
 */
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

/**
 * @brief Registers canonicalization patterns for XXMinusYYOp.
 *
 * Adds the MergeSubsequentXXMinusYY rewrite pattern into the provided pattern
 * set so that adjacent XXMinusYYOp instances acting on the same qubits can
 * be merged.
 *
 * @param results Pattern set to which canonicalization patterns will be added.
 * @param context MLIR context used to construct the patterns.
 */
void XXMinusYYOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<MergeSubsequentXXMinusYY>(context);
}

/**
 * @brief Computes the 4x4 unitary matrix for this XX-YY operation when `theta`
 * and `beta` are available as doubles.
 *
 * If either `theta` or `beta` cannot be converted to a double, the function
 * returns `std::nullopt`.
 *
 * @return std::optional<Eigen::Matrix4cd> The unitary matrix with rows:
 * Row 0: {mc, 0, 0, msm}
 * Row 1: {0, 1, 0, 0}
 * Row 2: {0, 0, 1, 0}
 * Row 3: {msp, 0, 0, mc}
 *
 * where mc = cos(theta/2), s = sin(theta/2),
 * msp = polar(s, beta - pi/2), and msm = polar(s, -beta - pi/2).
 */
std::optional<Eigen::Matrix4cd> XXMinusYYOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  const auto theta = valueToDouble(getTheta());
  const auto beta = valueToDouble(getBeta());
  if (!theta || !beta) {
    return std::nullopt;
  }

  const auto m0 = 0.0 + 0i;
  const auto m1 = 1.0 + 0i;
  const auto mc = std::cos(*theta / 2.0) + 0i;
  const auto s = std::sin(*theta / 2.0);
  const auto msp = std::polar(s, *beta - (std::numbers::pi / 2.));
  const auto msm = std::polar(s, -*beta - (std::numbers::pi / 2.));
  return Eigen::Matrix4cd{{mc, m0, m0, msm},  // row 0
                          {m0, m1, m0, m0},   // row 1
                          {m0, m0, m1, m0},   // row 2
                          {msp, m0, m0, mc}}; // row 3
}
