/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <complex>
#include <optional>
#include <variant>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::utils;

namespace {

/**
 * @brief Merge subsequent RZZ operations on the same qubits by adding their
 * angles.
 */
struct MergeSubsequentRZZ final : OpRewritePattern<RZZOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RZZOp op,
                                PatternRewriter& rewriter) const override {
    return mergeTwoTargetOneParameter(op, rewriter, true);
  }
};

} // namespace

void RZZOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  Value qubit0In, Value qubit1In,
                  const std::variant<double, Value>& theta) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand);
}

LogicalResult RZZOp::fold(FoldAdaptor /*adaptor*/,
                          SmallVectorImpl<OpFoldResult>& results) {
  if (const auto theta = valueToDouble(getTheta());
      theta && std::abs(*theta) <= TOLERANCE) {
    results.emplace_back(getInputQubit(0));
    results.emplace_back(getInputQubit(1));
    return success();
  }
  return failure();
}

void RZZOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<MergeSubsequentRZZ>(context);
}

std::optional<Matrix4x4> RZZOp::getUnitaryMatrix() {
  if (const auto theta = valueToDouble(getTheta())) {
    const auto mp = std::polar(1.0, *theta / 2);
    const auto mm = std::polar(1.0, -*theta / 2);
    return Matrix4x4::fromElements(mm, 0, 0, 0,  // row 0
                                   0, mp, 0, 0,  // row 1
                                   0, 0, mp, 0,  // row 2
                                   0, 0, 0, mm); // row 3
  }
  return std::nullopt;
}
