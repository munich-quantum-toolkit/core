/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Support/MQT/ConstantFolding.h"

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>

namespace mlir::mqt {

/// Largest supported magnitude of a global-phase angle in radians.
inline constexpr double MAX_GLOBAL_PHASE_ANGLE = 1.0e4;

/// Check the compiler-wide global-phase angle contract.
[[nodiscard]] inline bool isValidGlobalPhaseAngle(const double theta) {
  return std::isfinite(theta) && std::abs(theta) <= MAX_GLOBAL_PHASE_ANGLE;
}

/// Verify the compiler-wide global-phase angle contract.
[[nodiscard]] inline LogicalResult verifyGlobalPhaseAngle(Operation* operation,
                                                          const Value angle) {
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
