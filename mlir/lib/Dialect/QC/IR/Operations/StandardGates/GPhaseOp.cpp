/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/Utils/Angles.h"
#include "mlir/Dialect/MQT/Utils/Parameters.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <variant>

using namespace mlir;
using namespace mlir::qc;
using namespace mlir::mqt;

void GPhaseOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                     const std::variant<double, Value>& theta) {
  auto thetaOperand = variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, thetaOperand);
}

LogicalResult GPhaseOp::verify() {
  return verifyGlobalPhaseAngle(getOperation(), getTheta());
}
