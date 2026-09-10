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

void XXMinusYYOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                        Value qubit0In, Value qubit1In,
                        const std::variant<double, Value>& theta,
                        const std::variant<double, Value>& beta) {
  auto thetaOperand = variantToValue(odsBuilder, odsState.location, theta);
  auto betaOperand = variantToValue(odsBuilder, odsState.location, beta);
  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand, betaOperand);
}

LogicalResult XXMinusYYOp::fold(FoldAdaptor /*adaptor*/,
                                SmallVectorImpl<OpFoldResult>& results) {
  if (const auto theta = valueToDouble(getTheta());
      theta && std::abs(*theta) <= PARAMETER_COMPARISON_TOLERANCE) {
    results.emplace_back(getInputQubit(0));
    results.emplace_back(getInputQubit(1));
    return success();
  }
  return failure();
}

void XXMinusYYOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* /*context*/) {
  results.add(&mergeXXPlusMinusYY<XXMinusYYOp>);
}

Matrix4x4 XXMinusYYOp::unitaryMatrix(const double theta, const double beta) {
  using namespace std::complex_literals;
  const auto mc = std::cos(theta / 2);
  const auto s = std::sin(theta / 2);
  const auto msp = -1i * s * std::exp(1i * beta);
  const auto msm = -std::conj(msp);
  return Matrix4x4::fromElements(mc, 0, 0, msm,  // row 0
                                 0, 1, 0, 0,     // row 1
                                 0, 0, 1, 0,     // row 2
                                 msp, 0, 0, mc); // row 3
}

std::optional<Matrix4x4> XXMinusYYOp::getUnitaryMatrix() {
  const auto theta = valueToDouble(getTheta());
  const auto beta = valueToDouble(getBeta());
  if (!theta || !beta) {
    return std::nullopt;
  }
  return unitaryMatrix(*theta, *beta);
}
