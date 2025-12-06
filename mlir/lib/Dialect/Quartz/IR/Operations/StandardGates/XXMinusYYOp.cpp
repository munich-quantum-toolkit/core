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

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OperationSupport.h>
#include <variant>

using namespace mlir;
using namespace mlir::quartz;

void XXMinusYYOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                        const Value qubit0In, const Value qubit1In,
                        const std::variant<double, Value>& theta,
                        const std::variant<double, Value>& beta) {
  Value thetaOperand;
  if (std::holds_alternative<double>(theta)) {
    thetaOperand = odsBuilder.create<arith::ConstantOp>(
        odsState.location, odsBuilder.getF64FloatAttr(std::get<double>(theta)));
  } else {
    thetaOperand = std::get<Value>(theta);
  }

  Value betaOperand;
  if (std::holds_alternative<double>(beta)) {
    betaOperand = odsBuilder.create<arith::ConstantOp>(
        odsState.location, odsBuilder.getF64FloatAttr(std::get<double>(beta)));
  } else {
    betaOperand = std::get<Value>(beta);
  }

  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand, betaOperand);
}
