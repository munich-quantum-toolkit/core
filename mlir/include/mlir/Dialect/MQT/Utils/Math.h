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

#include <cmath>
#include <numbers>

namespace mlir::mqt {

/// Check whether a floating-point value is an integer.
[[nodiscard]] inline bool isIntegerExponent(const double value) {
  return value == std::floor(value) && std::isfinite(value);
}

/// Check whether a floating-point value is an even integer.
[[nodiscard]] inline bool isEvenExponent(const double value) {
  return std::fmod(std::fabs(value), 2.0) == 0.0;
}

/// Normalize an angle to (-pi, pi].
[[nodiscard]] inline double normalizeAngle(double theta) {
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

/// Default absolute tolerance for angle wrapping and phase-zero checks.
inline constexpr double TOLERANCE = 1e-15;

} // namespace mlir::mqt
