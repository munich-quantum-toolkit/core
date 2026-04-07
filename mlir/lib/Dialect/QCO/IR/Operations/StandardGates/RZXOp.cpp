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
 * @brief Merge subsequent RZX operations on the same qubits by adding their
 * angles.
 */
struct MergeSubsequentRZX final : OpRewritePattern<RZXOp> {
  using OpRewritePattern::OpRewritePattern;

  /**
   * Attempt to merge consecutive RZX operations acting on the same two qubits by
   * combining their rotation parameters and rewriting the IR accordingly.
   *
   * @param op The RZX operation to consider for merging.
   * @param rewriter Utility used to perform IR replacements during the rewrite.
   * @return LogicalResult `success()` if a merge and rewrite were performed, `failure()` otherwise.
   */
  LogicalResult matchAndRewrite(RZXOp op,
                                PatternRewriter& rewriter) const override {
    return mergeTwoTargetOneParameter(op, rewriter);
  }
};

} /**
 * @brief Construct an RZX operation, accepting either a double angle or an SSA value for theta.
 *
 * Converts the provided `theta` variant into an operand value and uses it to populate
 * the given OperationState via the OpBuilder.
 *
 * @param odsBuilder Builder used to create the operation.
 * @param odsState OperationState to populate for the new RZX operation.
 * @param qubit0In Value representing the first input qubit.
 * @param qubit1In Value representing the second input qubit.
 * @param theta Angle parameter for the RZX gate, given either as a double or as a Value.
 */

void RZXOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  Value qubit0In, Value qubit1In,
                  const std::variant<double, Value>& theta) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand);
}

/**
 * @brief Fold the RZX operation away when its rotation angle is effectively zero.
 *
 * When the operation's `theta` operand is a constant whose absolute value is
 * less than or equal to `TOLERANCE`, replaces the operation by emitting its
 * two input qubit values into `results`.
 *
 * @param results Output vector that will be populated with the two input
 *                qubit values when folding succeeds.
 * @return LogicalResult `success()` if the operation was folded and `results`
 *         contains the two input qubits, `failure()` otherwise.
 */
LogicalResult RZXOp::fold(FoldAdaptor /*adaptor*/,
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
 * @brief Register canonicalization patterns for RZXOp.
 *
 * Adds the MergeSubsequentRZX rewrite pattern to the provided pattern set so
 * that consecutive RZX operations targeting the same qubits can be merged.
 *
 * @param results Pattern set to populate with canonicalization patterns.
 * @param context MLIR context used to construct the pattern.
 */
void RZXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<MergeSubsequentRZX>(context);
}

/**
 * Compute the 4×4 unitary matrix of the RZX gate for the operation's theta parameter.
 *
 * If the operation's `theta` can be converted to a numeric value, the returned matrix is
 * constructed using mc = cos(theta / 2) (real) and ms = i * sin(theta / 2) (pure imaginary),
 * with rows:
 *   [ mc, -ms,  0,  0 ]
 *   [ -ms, mc,  0,  0 ]
 *   [  0,   0, mc, ms ]
 *   [  0,   0, ms, mc ]
 *
 * @return `std::optional<Eigen::Matrix4cd>` containing the 4×4 complex unitary matrix when
 *         `theta` is available; `std::nullopt` otherwise.
 */
std::optional<Eigen::Matrix4cd> RZXOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  if (const auto theta = valueToDouble(getTheta())) {
    const auto m0 = 0i;
    const auto mc = std::complex<double>{std::cos(*theta / 2.0)};
    const auto ms = std::complex<double>{0.0, std::sin(*theta / 2.0)};
    return Eigen::Matrix4cd{{mc, -ms, m0, m0}, // row 0
                            {-ms, mc, m0, m0}, // row 1
                            {m0, m0, mc, ms},  // row 2
                            {m0, m0, ms, mc}}; // row 3
  }
  return std::nullopt;
}
