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

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <optional>

namespace mlir::mqt {

/// Largest supported magnitude of a global-phase angle in radians.
inline constexpr double MAX_GLOBAL_PHASE_ANGLE = 1.0e4;

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
enum class PhaseGate { Identity, Z, S, Sdg, T, Tdg };

/// Classify a finite phase angle modulo 2*pi using the parameter tolerance.
///
/// Return no value when the angle needs a general P gate.
[[nodiscard]] std::optional<PhaseGate> classifyPhaseGate(double angle);

/// Normalize an angle to (-pi, pi].
[[nodiscard]] double normalizeAngle(double theta);

/// Check the compiler-wide global-phase angle contract.
[[nodiscard]] bool isValidGlobalPhaseAngle(double theta);

/// Verify the compiler-wide global-phase angle contract.
[[nodiscard]] LogicalResult verifyGlobalPhaseAngle(Operation* operation,
                                                   Value angle);

} // namespace mlir::mqt
