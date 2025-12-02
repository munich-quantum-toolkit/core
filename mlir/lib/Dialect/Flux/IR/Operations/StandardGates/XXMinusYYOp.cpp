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

namespace {

/**
 * @brief Merge subsequent XXMinusYY operations on the same qubits by adding
 * their thetas.
 */
struct MergeSubsequentXXMinusYY final : OpRewritePattern<XXMinusYYOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(XXMinusYYOp op,
                                PatternRewriter& rewriter) const override {
    auto prevOp = op.getQubit0In().getDefiningOp<XXMinusYYOp>();
    if (!prevOp) {
      return failure();
    }

    // Confirm operations act on same qubits
    if (op.getQubit1In() != prevOp.getQubit1Out()) {
      return failure();
    }

    // Confirm betas are equal
    auto beta = XXMinusYYOp::getStaticParameter(op.getBeta());
    auto prevBeta = XXMinusYYOp::getStaticParameter(prevOp.getBeta());
    if (beta && prevBeta) {
      if (beta.getValueAsDouble() != prevBeta.getValueAsDouble()) {
        return failure();
      }
    } else {
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

void XXMinusYYOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                        const Value qubit0In, const Value qubit1In,
                        const std::variant<double, Value>& theta,
                        const std::variant<double, Value>& beta) {
  Value thetaOperand = nullptr;
  if (std::holds_alternative<double>(theta)) {
    thetaOperand = odsBuilder.create<arith::ConstantOp>(
        odsState.location, odsBuilder.getF64FloatAttr(std::get<double>(theta)));
  } else {
    thetaOperand = std::get<Value>(theta);
  }

  Value betaOperand = nullptr;
  if (std::holds_alternative<double>(beta)) {
    betaOperand = odsBuilder.create<arith::ConstantOp>(
        odsState.location, odsBuilder.getF64FloatAttr(std::get<double>(beta)));
  } else {
    betaOperand = std::get<Value>(beta);
  }

  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand, betaOperand);
}

void XXMinusYYOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<MergeSubsequentXXMinusYY>(context);
}
