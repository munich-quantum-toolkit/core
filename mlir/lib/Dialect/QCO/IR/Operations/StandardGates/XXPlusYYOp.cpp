/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <Eigen/Core>
#include <cmath>
#include <complex>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
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

/**
 * @brief Remove trivial XXPlusYY operations.
 */
struct RemoveTrivialXXPlusYY final : OpRewritePattern<XXPlusYYOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(XXPlusYYOp op,
                                PatternRewriter& rewriter) const override {
    return removeTrivialTwoTargetOneParameter(op, rewriter);
  }
};

} // namespace

void XXPlusYYOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                       Value qubit0In, Value qubit1In,
                       const std::variant<double, Value>& theta,
                       const std::variant<double, Value>& beta) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  const auto betaOperand = variantToValue(odsBuilder, odsState.location, beta);
  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand, betaOperand);
}

void XXPlusYYOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.add<MergeSubsequentXXPlusYY, RemoveTrivialXXPlusYY>(context);
}

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
