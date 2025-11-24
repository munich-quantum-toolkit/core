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
#include "mlir/Dialect/Utils/MatrixUtils.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <numbers>
#include <variant>

using namespace mlir;
using namespace mlir::flux;
using namespace mlir::utils;

namespace {

/**
 * @brief Remove trivial U2 operations.
 */
struct RemoveSubsequentU2 final : OpRewritePattern<U2Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(U2Op op,
                                PatternRewriter& rewriter) const override {
    const auto phi = op.getStaticParameter(op.getPhi());
    const auto lambda = op.getStaticParameter(op.getLambda());
    if (!phi || !lambda) {
      return failure();
    }

    const auto phiValue = phi.getValueAsDouble();
    const auto lambdaValue = lambda.getValueAsDouble();
    if (phiValue != 0.0 || lambdaValue != 0.0) {
      return failure();
    }

    auto rxOp = rewriter.create<RYOp>(op.getLoc(), op.getQubitIn(),
                                      std::numbers::pi / 2.0);
    rewriter.replaceOp(op, rxOp.getResult());

    return success();
  }
};

} // namespace

DenseElementsAttr U2Op::tryGetStaticMatrix() {
  const auto phi = getStaticParameter(getPhi());
  const auto lambda = getStaticParameter(getLambda());
  if (!phi || !lambda) {
    return nullptr;
  }
  const auto phiValue = phi.getValueAsDouble();
  const auto lambdaValue = lambda.getValueAsDouble();
  return getMatrixU2(getContext(), phiValue, lambdaValue);
}

void U2Op::build(OpBuilder& odsBuilder, OperationState& odsState,
                 const Value qubitIn, const std::variant<double, Value>& phi,
                 const std::variant<double, Value>& lambda) {
  Value phiOperand = nullptr;
  if (std::holds_alternative<double>(phi)) {
    phiOperand = odsBuilder.create<arith::ConstantOp>(
        odsState.location, odsBuilder.getF64FloatAttr(std::get<double>(phi)));
  } else {
    phiOperand = std::get<Value>(phi);
  }

  Value lambdaOperand = nullptr;
  if (std::holds_alternative<double>(lambda)) {
    lambdaOperand = odsBuilder.create<arith::ConstantOp>(
        odsState.location,
        odsBuilder.getF64FloatAttr(std::get<double>(lambda)));
  } else {
    lambdaOperand = std::get<Value>(lambda);
  }

  build(odsBuilder, odsState, qubitIn, phiOperand, lambdaOperand);
}

void U2Op::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<RemoveSubsequentU2>(context);
}
