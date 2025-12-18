/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <variant>

using namespace mlir;
using namespace mlir::qc;
using namespace mlir::utils;

void UOp::build(OpBuilder& builder, OperationState& state, Value qubitIn,
                const std::variant<double, Value>& theta,
                const std::variant<double, Value>& phi,
                const std::variant<double, Value>& lambda) {
  auto thetaOperand = variantToValue(builder, state, theta);
  auto phiOperand = variantToValue(builder, state, phi);
  auto lambdaOperand = variantToValue(builder, state, lambda);
  build(builder, state, qubitIn, thetaOperand, phiOperand, lambdaOperand);
}
