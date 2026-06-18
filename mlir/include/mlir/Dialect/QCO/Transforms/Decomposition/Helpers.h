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

#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <complex>

/// Numeric helpers used by the decomposition passes.

namespace mlir::qco::helpers {

/// Check whether `matrix` is unitary within `tolerance` (i.e. `M^H M` is
/// approximately the identity).
[[nodiscard]] bool isUnitaryMatrix(const Matrix2x2& matrix,
                                   double tolerance = 1e-12);

/// Check whether `matrix` is unitary within `tolerance` (i.e. `M^H M` is
/// approximately the identity).
[[nodiscard]] bool isUnitaryMatrix(const Matrix4x4& matrix,
                                   double tolerance = 1e-12);

/**
 * Euclidean remainder of a modulo b.
 * The returned value is never negative.
 */
[[nodiscard]] double remEuclid(double a, double b);

/**
 * Wrap angle into interval [-pi, pi). If within atol of the endpoint, clamp to
 * -pi.
 */
[[nodiscard]] double mod2pi(double angle, double angleZeroEpsilon = 1e-13);

/**
 * Convert a two-qubit trace overlap into the average gate fidelity metric used
 * by the decomposition cost code.
 */
[[nodiscard]] double traceToFidelity(const std::complex<double>& x);

/**
 * Return the scalar `e^(i * globalPhase)` factor for a stored global phase.
 */
[[nodiscard]] std::complex<double> globalPhaseFactor(double globalPhase);

} // namespace mlir::qco::helpers
