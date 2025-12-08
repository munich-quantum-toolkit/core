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
#include "mlir/Dialect/Utils/Utils.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OperationSupport.h>
#include <variant>

using namespace mlir;
using namespace mlir::quartz;
using namespace mlir::utils;

void RXXOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  const Value qubit0In, const Value qubit1In,
                  const std::variant<double, Value>& theta) {
  const auto& thetaOperand = variantToValue(odsBuilder, odsState, theta);
  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand);
}
