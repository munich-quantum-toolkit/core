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
#include "mlir/Dialect/Utils/Utils.h"

#include <Eigen/Core>
#include <cmath>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
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

  LogicalResult matchAndRewrite(XXMinusYYOp op,
                                PatternRewriter& rewriter) const override {
    auto prevOp = op.getInputQubit(0).getDefiningOp<XXMinusYYOp>();
    if (!prevOp) {
      return failure();
    }

    // Confirm operations act on same qubits
    if (op.getInputQubit(1) != prevOp.getOutputQubit(1)) {
      return failure();
    }

    // Confirm betas are equal
    auto beta = valueToDouble(op.getBeta());
    auto prevBeta = valueToDouble(prevOp.getBeta());
    if (beta && prevBeta) {
      if (std::abs(*beta - *prevBeta) > TOLERANCE) {
        return failure();
      }
    } else if (op.getBeta() != prevOp.getBeta()) {
      return failure();
    }

    // Compute and set new theta, which has index 2
    auto newParameter = rewriter.create<arith::AddFOp>(
        op.getLoc(), op.getOperand(2), prevOp.getOperand(2));
    op->setOperand(2, newParameter.getResult());

    // Trivialize predecessor
    rewriter.replaceOp(prevOp,
                       {prevOp.getInputQubit(0), prevOp.getInputQubit(1)});
    return success();
  }
};

} // namespace

void XXMinusYYOp::build(OpBuilder& builder, OperationState& state,
                        Value qubit0In, Value qubit1In,
                        const std::variant<double, Value>& theta,
                        const std::variant<double, Value>& beta) {
  auto thetaOperand = variantToValue(builder, state.location, theta);
  auto betaOperand = variantToValue(builder, state.location, beta);
  build(builder, state, qubit0In, qubit1In, thetaOperand, betaOperand);
}

void XXMinusYYOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<MergeSubsequentXXMinusYY>(context);
}

std::optional<Eigen::Matrix4cd> XXMinusYYOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  if (auto theta = utils::valueToDouble(getTheta())) {
    if (auto beta = utils::valueToDouble(getBeta())) {
      const auto m0 = 0.0 + 0i;
      const auto m1 = 1.0 + 0i;
      const auto mc = std::cos(*theta / 2.0) + 0i;
      const auto s = std::sin(*theta / 2.0);
      const auto msp = std::polar(s, -*beta);
      const auto msm = std::polar(-s, *beta);
      return Eigen::Matrix4cd{{mc, m0, m0, msm},  // row 0
                              {m0, m1, m0, m0},   // row 1
                              {m0, m0, m1, m0},   // row 2
                              {msp, m0, m0, mc}}; // row 3
    }
  }
  return std::nullopt;
}
