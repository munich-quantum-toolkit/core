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

#include "mlir/Dialect/MQT/Utils/ConstantFolding.h"

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <numbers>

namespace mlir::mqt {

double normalizeAngle(double theta) {
  const double twoPi = 2.0 * std::numbers::pi;
  theta = std::fmod(theta, twoPi);
  if (theta > std::numbers::pi) {
    theta -= twoPi;
  }
  if (theta <= -std::numbers::pi) {
    theta += twoPi;
  }
  return theta;
}

bool isValidGlobalPhaseAngle(const double theta) {
  return std::isfinite(theta) && std::abs(theta) <= MAX_GLOBAL_PHASE_ANGLE;
}

LogicalResult verifyGlobalPhaseAngle(Operation* operation, Value angle) {
  const auto constant = valueToConstantDouble(angle);
  if (!constant || !std::isfinite(*constant)) {
    return success();
  }
  if (!isValidGlobalPhaseAngle(*constant)) {
    return operation->emitOpError()
           << "constant angle must have magnitude at most "
           << MAX_GLOBAL_PHASE_ANGLE << " radians";
  }
  return success();
}

} // namespace mlir::mqt
