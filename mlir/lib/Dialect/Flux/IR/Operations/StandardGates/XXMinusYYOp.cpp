/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Flux/IR/FluxDialect.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <cmath>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <variant>

using namespace mlir;
using namespace mlir::flux;
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
    auto beta = XXMinusYYOp::getStaticParameter(op.getBeta());
    auto prevBeta = XXMinusYYOp::getStaticParameter(prevOp.getBeta());
    if (beta && prevBeta) {
      if (std::abs(beta.getValueAsDouble() - prevBeta.getValueAsDouble()) >
          TOLERANCE) {
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
  auto thetaOperand = variantToValue(builder, state, theta);
  auto betaOperand = variantToValue(builder, state, beta);
  build(builder, state, qubit0In, qubit1In, thetaOperand, betaOperand);
}

void XXMinusYYOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<MergeSubsequentXXMinusYY>(context);
}
