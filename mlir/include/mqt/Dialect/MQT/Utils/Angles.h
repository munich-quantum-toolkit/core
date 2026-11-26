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

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <numbers>
#include <optional>
#include <utility>

namespace mlir::mqt {

/// Largest supported magnitude of a global-phase angle in radians.
inline constexpr double MAX_GLOBAL_PHASE_ANGLE = 1.0e4;

/// Return `lhs + rhs` and the exact rounding error of that sum (TwoSum).
///
/// The error is exact even when one operand is much smaller than the other.
[[nodiscard]] constexpr std::pair<double, double> twoSum(const double lhs,
                                                         const double rhs) {
  const double sum = lhs + rhs;
  const double rhsRounded = sum - lhs;
  return {sum, (lhs - (sum - rhsRounded)) + (rhs - rhsRounded)};
}

/// Add finite constant angles when the absolute rounding error is no greater
/// than PARAMETER_COMPARISON_TOLERANCE.
///
/// Return no value on overflow or excess error. This bounds a rewrite's
/// arithmetic, not the accepted input angles.
[[nodiscard]] std::optional<double> addConstantAngles(double lhs, double rhs);

/// Multiply finite constants with the same error bound as addConstantAngles.
[[nodiscard]] std::optional<double> scaleConstantAngle(double angle,
                                                       double factor);

/// Named gates represented by a constant P-gate angle.
enum class PhaseGate : uint8_t { Identity, Z, S, Sdg, T, Tdg };

/// Classify a finite phase angle modulo 2*π using the parameter tolerance.
///
/// Return no value when the angle needs a general P gate.
[[nodiscard]] std::optional<PhaseGate> classifyPhaseGate(double angle);

/// Reduce `angle` modulo `period` to (-period/2, period/2].
///
/// `period` must be a power-of-two multiple of π, such as 2π for phases or 4π
/// for rotations. libm reduces the argument with an extended-precision π, so
/// large angles stay accurate where `fmod` by binary64 π does not.
[[nodiscard]] double reduceAngle(double angle, double period);

/// Normalize a phase angle to (-π, π].
[[nodiscard]] inline double normalizeAngle(const double theta) {
  return reduceAngle(theta, 2.0 * std::numbers::pi);
}

/// Check the compiler-wide global-phase angle contract.
[[nodiscard]] bool isValidGlobalPhaseAngle(double theta);

/// Sanity-check direct constants against the global-phase angle contract.
[[nodiscard]] LogicalResult verifyGlobalPhaseAngle(Operation* operation,
                                                   Value angle);

} // namespace mlir::mqt
