/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Flux/FluxUtils.h"
#include "mlir/Dialect/Flux/IR/FluxDialect.h"
#include "mlir/Dialect/Utils/Utils.h"

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
 * @brief Merge subsequent XXPlusYY operations on the same qubits by adding
 * their thetas.
 */
struct MergeSubsequentXXPlusYY final : OpRewritePattern<XXPlusYYOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(XXPlusYYOp op,
                                PatternRewriter& rewriter) const override {
    auto prevOp = op.getQubit0In().getDefiningOp<XXPlusYYOp>();
    if (!prevOp) {
      return failure();
    }

    // Confirm operations act on same qubits
    if (op.getQubit1In() != prevOp.getQubit1Out()) {
      return failure();
    }

    // Confirm betas are equal
    auto beta = XXPlusYYOp::getStaticParameter(op.getBeta());
    auto prevBeta = XXPlusYYOp::getStaticParameter(prevOp.getBeta());
    if (beta && prevBeta) {
      if (std::abs(beta.getValueAsDouble() - prevBeta.getValueAsDouble()) >
          TOLERANCE) {
        return failure();
      }
    } else if (op.getBeta() != prevOp.getBeta()) {
      return failure();
    }

    // Compute and set new theta
    auto newParameter = rewriter.create<arith::AddFOp>(
        op.getLoc(), op.getOperand(2), prevOp.getOperand(2));
    op->setOperand(2, newParameter.getResult());

    // Trivialize predecessor
    rewriter.replaceOp(prevOp, {prevOp.getQubit0In(), prevOp.getQubit1In()});

    return success();
  }
};

} // namespace

void XXPlusYYOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                       const Value qubit0In, const Value qubit1In,
                       const std::variant<double, Value>& theta,
                       const std::variant<double, Value>& beta) {
  const auto& thetaOperand = variantToValue(odsBuilder, odsState, theta);
  const auto& betaOperand = variantToValue(odsBuilder, odsState, beta);
  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand, betaOperand);
}

void XXPlusYYOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.add<MergeSubsequentXXPlusYY>(context);
}
