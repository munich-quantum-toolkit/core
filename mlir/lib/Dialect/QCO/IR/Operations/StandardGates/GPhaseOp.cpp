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
#include <complex>
#include <mlir/IR/Builders.h>
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
 * @brief Remove trivial GPhase operations.
 */
struct RemoveTrivialGPhase final : OpRewritePattern<GPhaseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GPhaseOp op,
                                PatternRewriter& rewriter) const override {
    if (const auto theta = valueToDouble(op.getTheta());
        !theta || std::abs(*theta) > TOLERANCE) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void GPhaseOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                     const std::variant<double, Value>& theta) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, thetaOperand);
}

void GPhaseOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<RemoveTrivialGPhase>(context);
}

std::optional<Eigen::Matrix<std::complex<double>, 1, 1>>
GPhaseOp::getUnitaryMatrix() {
  if (const auto theta = valueToDouble(getTheta())) {
    return Eigen::Matrix<std::complex<double>, 1, 1>{std::polar(1.0, *theta)};
  }
  return std::nullopt;
}
