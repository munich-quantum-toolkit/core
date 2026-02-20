/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <variant>

using namespace mlir;
using namespace mlir::qc;
using namespace mlir::utils;

void RZZOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  Value qubit0In, Value qubit1In,
                  const std::variant<double, Value>& theta) {
  auto thetaOperand = variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, qubit0In, qubit1In, thetaOperand);
}
