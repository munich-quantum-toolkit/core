/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <cmath>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <variant>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::utils;

namespace {

/**
 * @brief Remove trivial GPhase operations.
 */
struct RemoveTrivialGPhase final : OpRewritePattern<GPhaseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GPhaseOp op,
                                PatternRewriter& rewriter) const override {
    const auto thetaAttr = GPhaseOp::getStaticParameter(op.getTheta());
    if (!thetaAttr) {
      return failure();
    }

    const auto thetaValue = thetaAttr.getValueAsDouble();
    if (std::abs(thetaValue) > TOLERANCE) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void GPhaseOp::build(OpBuilder& builder, OperationState& state,
                     const std::variant<double, Value>& theta) {
  auto thetaOperand = variantToValue(builder, state, theta);
  build(builder, state, thetaOperand);
}

void GPhaseOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<RemoveTrivialGPhase>(context);
}
