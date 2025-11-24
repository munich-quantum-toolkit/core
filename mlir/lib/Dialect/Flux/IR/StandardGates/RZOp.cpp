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
#include <variant>

using namespace mlir;
using namespace mlir::flux;
using namespace mlir::utils;

namespace {

/**
 * @brief Merge subsequent RZ operations on the same qubit by adding their
 * angles.
 */
struct MergeSubsequentRZ final : OpRewritePattern<RZOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RZOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is an RZOp
    auto prevOp = op.getQubitIn().getDefiningOp<RZOp>();
    if (!prevOp) {
      return failure();
    }

    // Compute and set new theta
    auto newTheta = rewriter.create<arith::AddFOp>(op.getLoc(), op.getTheta(),
                                                   prevOp.getTheta());
    op->setOperand(1, newTheta.getResult());

    // Trivialize previous RZOp
    rewriter.replaceOp(prevOp, prevOp.getQubitIn());

    return success();
  }
};

} // namespace

DenseElementsAttr RZOp::tryGetStaticMatrix() {
  const auto& theta = getStaticParameter(getTheta());
  if (!theta) {
    return nullptr;
  }
  const auto thetaValue = theta.getValueAsDouble();
  return getMatrixRZ(getContext(), thetaValue);
}

void RZOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                 const Value qubitIn,
                 const std::variant<double, Value>& theta) {
  Value thetaOperand = nullptr;
  if (std::holds_alternative<double>(theta)) {
    thetaOperand = odsBuilder.create<arith::ConstantOp>(
        odsState.location, odsBuilder.getF64FloatAttr(std::get<double>(theta)));
  } else {
    thetaOperand = std::get<Value>(theta);
  }
  build(odsBuilder, odsState, qubitIn, thetaOperand);
}

void RZOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<MergeSubsequentRZ>(context);
}
