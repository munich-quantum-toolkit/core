/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Utils/Matrix.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include <cmath>
#include <complex>
#include <optional>
#include <variant>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::mqt;

void RZXOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  Value qubit0In, Value qubit1In,
                  const std::variant<double, Value>& theta) {
  auto thetaOperand = variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand);
}

LogicalResult RZXOp::fold(FoldAdaptor /*adaptor*/,
                          SmallVectorImpl<OpFoldResult>& results) {
  if (const auto theta = valueToDouble(getTheta());
      theta && std::abs(*theta) <= PARAMETER_COMPARISON_TOLERANCE) {
    results.emplace_back(getInputQubit(0));
    results.emplace_back(getInputQubit(1));
    return success();
  }
  return failure();
}

void RZXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* /*context*/) {
  results.add(+[](RZXOp op, PatternRewriter& rewriter) {
    return mergeTwoTargetOneParameter(op, rewriter);
  });
}

Matrix4x4 RZXOp::unitaryMatrix(const double theta) {
  using namespace std::complex_literals;

  const auto mc = std::cos(theta / 2);
  const auto ms = 1i * std::sin(theta / 2);
  return Matrix4x4::fromElements(mc, -ms, 0, 0, // row 0
                                 -ms, mc, 0, 0, // row 1
                                 0, 0, mc, ms,  // row 2
                                 0, 0, ms, mc); // row 3
}

std::optional<Matrix4x4> RZXOp::getUnitaryMatrix() {
  if (const auto theta = valueToDouble(getTheta())) {
    return unitaryMatrix(*theta);
  }
  return std::nullopt;
}
