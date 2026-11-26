/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Utils/Angles.h"

#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include <cmath>
#include <numbers>
#include <optional>

namespace mlir::mqt {

std::optional<double> addConstantAngles(double lhs, double rhs) {
  const auto [sum, error] = twoSum(lhs, rhs);
  if (!std::isfinite(sum) || !std::isfinite(error) ||
      std::abs(error) > PARAMETER_COMPARISON_TOLERANCE) {
    return std::nullopt;
  }
  return sum;
}

std::optional<double> scaleConstantAngle(double angle, double factor) {
  const double product = angle * factor;
  if (!std::isfinite(product) || std::abs(std::fma(angle, factor, -product)) >
                                     PARAMETER_COMPARISON_TOLERANCE) {
    return std::nullopt;
  }
  return product;
}

std::optional<PhaseGate> classifyPhaseGate(double angle) {
  const double normalized = normalizeAngle(angle);
  const double pi = std::numbers::pi;
  const auto near = [&](double target) {
    return std::abs(normalized - target) < PARAMETER_COMPARISON_TOLERANCE;
  };
  if (near(0.0)) {
    return PhaseGate::Identity;
  }
  // normalizeAngle maps -π to π, but -π + ε stays near -π.
  if (near(pi) || near(-pi)) {
    return PhaseGate::Z;
  }
  if (near(pi / 2.0)) {
    return PhaseGate::S;
  }
  if (near(-pi / 2.0)) {
    return PhaseGate::Sdg;
  }
  if (near(pi / 4.0)) {
    return PhaseGate::T;
  }
  if (near(-pi / 4.0)) {
    return PhaseGate::Tdg;
  }
  return std::nullopt;
}

double reduceAngle(double angle, const double period) {
  const double half = period / 2.0;
  if (std::abs(angle) > half) {
    // Scaling by a power of two is exact, and libm's tan reduces accurately.
    const double scale = period / std::numbers::pi;
    angle = scale * std::atan(std::tan(angle / scale));
  }
  return angle == -half ? half : angle;
}

bool isValidGlobalPhaseAngle(const double theta) {
  return std::abs(theta) <= MAX_GLOBAL_PHASE_ANGLE;
}

LogicalResult verifyGlobalPhaseAngle(Operation* operation, Value angle) {
  const auto constant = valueToDouble(angle);
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
