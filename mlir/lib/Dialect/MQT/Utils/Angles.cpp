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
#include "mlir/Dialect/MQT/Utils/Parameters.h"

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <numbers>
#include <optional>

namespace mlir::mqt {

std::optional<double> addConstantAngles(double lhs, double rhs) {
  const double sum = lhs + rhs;
  if (!std::isfinite(lhs) || !std::isfinite(rhs) || !std::isfinite(sum)) {
    return std::nullopt;
  }
  // TwoSum recovers the rounding error even when one operand is much smaller.
  const double rhsRounded = sum - lhs;
  const double error = (lhs - (sum - rhsRounded)) + (rhs - rhsRounded);
  if (!std::isfinite(error) ||
      std::abs(error) > PARAMETER_COMPARISON_TOLERANCE) {
    return std::nullopt;
  }
  return sum;
}

std::optional<double> scaleConstantAngle(double angle, double factor) {
  const double product = angle * factor;
  if (!std::isfinite(angle) || !std::isfinite(factor) ||
      !std::isfinite(product) ||
      std::abs(std::fma(angle, factor, -product)) >
          PARAMETER_COMPARISON_TOLERANCE) {
    return std::nullopt;
  }
  return product;
}

std::optional<PhaseGate> classifyPhaseGate(double angle) {
  if (!std::isfinite(angle)) {
    return std::nullopt;
  }
  const double normalized = normalizeAngle(angle);
  const double pi = std::numbers::pi;
  const auto validatedGate = [&](PhaseGate gate, double real,
                                 double imaginary) -> std::optional<PhaseGate> {
    // Reduction by binary64 pi can lose phase information for large angles.
    if (std::abs(std::cos(angle) - real) <= PARAMETER_COMPARISON_TOLERANCE &&
        std::abs(std::sin(angle) - imaginary) <=
            PARAMETER_COMPARISON_TOLERANCE) {
      return gate;
    }
    return std::nullopt;
  };
  if (std::abs(normalized) < PARAMETER_COMPARISON_TOLERANCE) {
    return validatedGate(PhaseGate::Identity, 1.0, 0.0);
  }
  if (std::abs(std::abs(normalized) - pi) < PARAMETER_COMPARISON_TOLERANCE) {
    return validatedGate(PhaseGate::Z, -1.0, 0.0);
  }
  if (std::abs(normalized - pi / 2.0) < PARAMETER_COMPARISON_TOLERANCE) {
    return validatedGate(PhaseGate::S, 0.0, 1.0);
  }
  if (std::abs(normalized + pi / 2.0) < PARAMETER_COMPARISON_TOLERANCE) {
    return validatedGate(PhaseGate::Sdg, 0.0, -1.0);
  }
  if (std::abs(normalized - pi / 4.0) < PARAMETER_COMPARISON_TOLERANCE) {
    const double component = std::sqrt(0.5);
    return validatedGate(PhaseGate::T, component, component);
  }
  if (std::abs(normalized + pi / 4.0) < PARAMETER_COMPARISON_TOLERANCE) {
    const double component = std::sqrt(0.5);
    return validatedGate(PhaseGate::Tdg, component, -component);
  }
  return std::nullopt;
}

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
