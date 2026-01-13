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
 * @brief Merge subsequent P operations on the same qubit by adding their
 * angles.
 */
struct MergeSubsequentP final : OpRewritePattern<POp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(POp op,
                                PatternRewriter& rewriter) const override {
    return mergeOneTargetOneParameter(op, rewriter);
  }
};

/**
 * @brief Remove trivial P operations.
 */
struct RemoveTrivialP final : OpRewritePattern<POp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(POp op,
                                PatternRewriter& rewriter) const override {
    return removeTrivialOneTargetOneParameter(op, rewriter);
  }
};

} // namespace

void POp::build(OpBuilder& builder, OperationState& state, Value qubitIn,
                const std::variant<double, Value>& theta) {
  auto thetaOperand = variantToValue(builder, state.location, theta);
  build(builder, state, qubitIn, thetaOperand);
}

void POp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<MergeSubsequentP, RemoveTrivialP>(context);
}

std::optional<Eigen::Matrix2cd> POp::getUnitaryMatrix() {
  if (auto theta = utils::valueToDouble(getTheta())) {
    return Eigen::Matrix2cd{{1.0, 0.0}, {0.0, std::polar(1.0, *theta)}};
  }
  return std::nullopt;
}
