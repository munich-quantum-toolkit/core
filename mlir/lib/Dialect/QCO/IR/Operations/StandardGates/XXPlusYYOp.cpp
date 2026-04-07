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
 * @brief Merge subsequent XXPlusYY operations on the same qubits by adding
 * their thetas.
 */
struct MergeSubsequentXXPlusYY final : OpRewritePattern<XXPlusYYOp> {
  using OpRewritePattern::OpRewritePattern;

  /**
   * @brief Attempts to merge an adjacent XXPlusYYOp into the provided operation.
   *
   * Matches when the single user of `op`'s output qubit 0 is another `XXPlusYYOp`
   * that acts on the same qubits and has an equivalent `beta` (within `TOLERANCE`
   * for numeric betas or exactly equal for non-constant betas). On success, it
   * replaces the second operation by updating the first operation's `theta`
   * operand (operand index 2) to the sum of both thetas and replaces the second
   * operation with the first operation's results.
   *
   * @param op The first `XXPlusYYOp` to match and potentially merge.
   * @param rewriter PatternRewriter used to create the new addition and perform replacements.
   * @returns `success()` if the operations were merged and rewritten, `failure()` otherwise.
   */
  LogicalResult matchAndRewrite(XXPlusYYOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the successor is the same operation
    auto nextOp =
        llvm::dyn_cast<XXPlusYYOp>(*op.getOutputQubit(0).user_begin());
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

} /**
 * @brief Builds an XXPlusYYOp from qubit inputs and variant theta/beta operands.
 *
 * Converts `theta` and `beta` (each either a `double` or an MLIR `Value`) into
 * operand `Value`s and forwards construction to the overload that accepts
 * `Value`-typed `theta` and `beta`.
 *
 * @param odsBuilder Builder used to create intermediate values and operations.
 * @param odsState Operation state being populated.
 * @param qubit0In First input qubit value (input/target for the gate).
 * @param qubit1In Second input qubit value (input/target for the gate).
 * @param theta Either a `double` rotation angle or an MLIR `Value` representing theta.
 * @param beta Either a `double` phase parameter or an MLIR `Value` representing beta.
 */

void XXPlusYYOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                       Value qubit0In, Value qubit1In,
                       const std::variant<double, Value>& theta,
                       const std::variant<double, Value>& beta) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  const auto betaOperand = variantToValue(odsBuilder, odsState.location, beta);
  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand, betaOperand);
}

/**
 * @brief Replaces the operation with its two input qubits when `theta` is a constant approximately zero.
 *
 * If `theta` can be converted to a double and its absolute value is less than or equal to
 * TOLERANCE, appends the two input-qubit values to `results` and returns success(); otherwise
 * leaves `results` unchanged and returns failure().
 *
 * @param results Container to receive fold results; on success it will contain the two input qubits.
 * @return LogicalResult `success()` if the operation was folded (results populated), `failure()` otherwise.
 */
LogicalResult XXPlusYYOp::fold(FoldAdaptor /*adaptor*/,
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
 * @brief Registers canonicalization rewrite patterns for XXPlusYYOp.
 *
 * Adds patterns to `results` that the canonicalizer will apply to simplify or
 * merge adjacent `XXPlusYYOp` operations.
 *
 * @param results Pattern set to which canonicalization patterns will be added.
 * @param context MLIR context used when constructing the rewrite patterns.
 */
void XXPlusYYOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.add<MergeSubsequentXXPlusYY>(context);
}

/**
 * @brief Computes the 4×4 unitary matrix of the XX+YY gate from the operation's `theta` and `beta` operands.
 *
 * If both `theta` and `beta` can be converted to concrete doubles, returns the complex 4×4 matrix
 * corresponding to the gate:
 *   [ [1, 0, 0, 0],
 *     [0, cos(theta/2), e^{i(beta - pi/2)} sin(theta/2), 0],
 *     [0, e^{-i(beta + pi/2)} sin(theta/2), cos(theta/2), 0],
 *     [0, 0, 0, 1] ]
 *
 * @return std::optional<Eigen::Matrix4cd> The unitary matrix when `theta` and `beta` are constants; `std::nullopt` if either value cannot be converted to a double.
 */
std::optional<Eigen::Matrix4cd> XXPlusYYOp::getUnitaryMatrix() {
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
  const auto msp = std::polar(s, *beta - (std::numbers::pi / 2));
  const auto msm = std::polar(s, -*beta - (std::numbers::pi / 2));
  return Eigen::Matrix4cd{{m1, m0, m0, m0},  // row 0
                          {m0, mc, msp, m0}, // row 1
                          {m0, msm, mc, m0}, // row 2
                          {m0, m0, m0, m1}}; // row 3
}
