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

} // namespace

void RZXOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  Value qubit0In, Value qubit1In,
                  const std::variant<double, Value>& theta) {
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand);
}

LogicalResult RZXOp::fold(FoldAdaptor /*adaptor*/,
                          SmallVectorImpl<OpFoldResult>& results) {
  if (const auto theta = valueToDouble(getTheta());
      theta && std::abs(*theta) <= TOLERANCE) {
    results.emplace_back(getInputQubit(0));
    results.emplace_back(getInputQubit(1));
    return success();
  }
  return failure();
}

void RZXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<MergeSubsequentRZX>(context);
}

std::optional<Matrix4x4> RZXOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  if (const auto theta = valueToDouble(getTheta())) {
    const auto mc = std::cos(*theta / 2);
    const auto ms = 1i * std::sin(*theta / 2);
    return Matrix4x4::fromElements(mc, -ms, 0, 0, // row 0
                                   -ms, mc, 0, 0, // row 1
                                   0, 0, mc, ms,  // row 2
                                   0, 0, ms, mc); // row 3
  }
  return std::nullopt;
}
