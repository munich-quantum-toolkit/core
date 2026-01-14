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
 * @brief Merge subsequent RZX operations on the same qubits by adding their
 * angles.
 */
struct MergeSubsequentRZX final : OpRewritePattern<RZXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RZXOp op,
                                PatternRewriter& rewriter) const override {
    return mergeTwoTargetOneParameter(op, rewriter);
  }
};

/**
 * @brief Remove trivial RZX operations.
 */
struct RemoveTrivialRZX final : OpRewritePattern<RZXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RZXOp op,
                                PatternRewriter& rewriter) const override {
    return removeTrivialTwoTargetOneParameter(op, rewriter);
  }
};

} // namespace

void RZXOp::build(OpBuilder& builder, OperationState& state, Value qubit0In,
                  Value qubit1In, const std::variant<double, Value>& theta) {
  auto thetaOperand = variantToValue(builder, state.location, theta);
  build(builder, state, qubit0In, qubit1In, thetaOperand);
}

void RZXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<MergeSubsequentRZX, RemoveTrivialRZX>(context);
}

std::optional<Eigen::Matrix4cd> RZXOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  if (auto theta = utils::valueToDouble(getTheta())) {
    const auto m0 = 0i;
    const auto mc = std::complex<double>{std::cos(*theta / 2.0)};
    const auto ms = std::complex<double>{0.0, std::sin(*theta / 2.0)};
    return Eigen::Matrix4cd{{mc, -ms, m0, m0}, // row 0
                            {-ms, mc, m0, m0}, // row 1
                            {m0, m0, mc, ms},  // row 2
                            {m0, m0, ms, mc}}; // row 3
  }
  return std::nullopt;
}
