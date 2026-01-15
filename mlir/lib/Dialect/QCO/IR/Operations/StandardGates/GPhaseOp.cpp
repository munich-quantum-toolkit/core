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

#include <cmath>
#include <mlir/IR/Builders.h>
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
    const auto theta = valueToDouble(op.getTheta());
    if (!theta || std::abs(*theta) > TOLERANCE) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void GPhaseOp::build(OpBuilder& builder, OperationState& state,
                     const std::variant<double, Value>& theta) {
  auto thetaOperand = variantToValue(builder, state.location, theta);
  build(builder, state, thetaOperand);
}

void GPhaseOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<RemoveTrivialGPhase>(context);
}
