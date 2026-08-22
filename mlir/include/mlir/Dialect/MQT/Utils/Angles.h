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

namespace mlir::mqt {

/// Largest supported magnitude of a global-phase angle in radians.
inline constexpr double MAX_GLOBAL_PHASE_ANGLE = 1.0e4;

/// Normalize an angle to (-pi, pi].
[[nodiscard]] double normalizeAngle(double theta);

/// Check the compiler-wide global-phase angle contract.
[[nodiscard]] bool isValidGlobalPhaseAngle(double theta);

/// Verify the compiler-wide global-phase angle contract.
[[nodiscard]] LogicalResult verifyGlobalPhaseAngle(Operation* operation,
                                                   Value angle);

} // namespace mlir::mqt
