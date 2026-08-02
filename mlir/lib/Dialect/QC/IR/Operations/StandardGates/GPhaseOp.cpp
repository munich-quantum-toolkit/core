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
#include <mlir/Support/LogicalResult.h>

#include <variant>

using namespace mlir;
using namespace mlir::qc;
using namespace mlir::utils;

void GPhaseOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                     const std::variant<double, Value>& theta) {
  auto thetaOperand = variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, thetaOperand);
}

LogicalResult GPhaseOp::verify() {
  const auto theta = valueToDouble(getTheta());
  if (theta && !isValidGlobalPhaseAngle(*theta)) {
    return emitOpError()
           << "constant angle must be finite and have magnitude at most "
           << MAX_GLOBAL_PHASE_ANGLE << " radians";
  }
  return success();
}
