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

#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <variant>

using namespace mlir;
using namespace mlir::quartz;
using namespace mlir::utils;

void ROp::build(OpBuilder& builder, OperationState& state, const Value qubitIn,
                const std::variant<double, Value>& theta,
                const std::variant<double, Value>& phi) {
  const auto& thetaOperand = variantToValue(builder, state, theta);
  const auto& phiOperand = variantToValue(builder, state, phi);
  build(builder, state, qubitIn, thetaOperand, phiOperand);
}
