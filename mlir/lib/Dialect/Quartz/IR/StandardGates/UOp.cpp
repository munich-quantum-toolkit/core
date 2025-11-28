/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"
#include "mlir/Dialect/Utils/MatrixUtils.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OperationSupport.h>
#include <variant>

using namespace mlir;
using namespace mlir::quartz;
using namespace mlir::utils;

DenseElementsAttr UOp::tryGetStaticMatrix() {
  const auto theta = getStaticParameter(getTheta());
  const auto phi = getStaticParameter(getPhi());
  const auto lambda = getStaticParameter(getLambda());
  if (!theta || !phi || !lambda) {
    return nullptr;
  }
  const auto thetaValue = theta.getValueAsDouble();
  const auto phiValue = phi.getValueAsDouble();
  const auto lambdaValue = lambda.getValueAsDouble();
  return getMatrixU(getContext(), thetaValue, phiValue, lambdaValue);
}

void UOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                const Value qubitIn, const std::variant<double, Value>& theta,
                const std::variant<double, Value>& phi,
                const std::variant<double, Value>& lambda) {
  Value thetaOperand = nullptr;
  if (std::holds_alternative<double>(theta)) {
    thetaOperand = odsBuilder.create<arith::ConstantOp>(
        odsState.location, odsBuilder.getF64FloatAttr(std::get<double>(theta)));
  } else {
    thetaOperand = std::get<Value>(theta);
  }

  Value phiOperand = nullptr;
  if (std::holds_alternative<double>(phi)) {
    phiOperand = odsBuilder.create<arith::ConstantOp>(
        odsState.location, odsBuilder.getF64FloatAttr(std::get<double>(phi)));
  } else {
    phiOperand = std::get<Value>(phi);
  }

  Value lambdaOperand = nullptr;
  if (std::holds_alternative<double>(lambda)) {
    lambdaOperand = odsBuilder.create<arith::ConstantOp>(
        odsState.location,
        odsBuilder.getF64FloatAttr(std::get<double>(lambda)));
  } else {
    lambdaOperand = std::get<Value>(lambda);
  }

  build(odsBuilder, odsState, qubitIn, thetaOperand, phiOperand, lambdaOperand);
}
