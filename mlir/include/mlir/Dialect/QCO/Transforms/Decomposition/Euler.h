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

#include <Eigen/Core>
#include <mlir/Dialect/Utils/Utils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>

namespace mlir::qco::decomposition {
/**
 * Target gate set for single-qubit Euler synthesis.
 *
 * - **KAK bases** (`ZYZ`, `ZXZ`, `XZX`, `XYX`): Euler decompositions named by
 *   the middle and outer rotation axes.
 * - **`U`**: single `u(theta, phi, lambda)` gate (angles from the Z-Y-Z form).
 * - **`ZSXX`**: `rz · sx · rz · sx · rz`, or `rz · x · rz` when the middle
 *   angle is ~0.
 */
enum class EulerBasis : std::uint8_t {
  ZYZ = 0,  ///< `rz · ry · rz`.
  ZXZ = 1,  ///< `rz · rx · rz`.
  XZX = 2,  ///< `rx · rz · rx`.
  XYX = 3,  ///< `rx · ry · rx`.
  U = 4,    ///< `u(theta, phi, lambda)`.
  ZSXX = 5, ///< `rz · sx · rz · sx · rz`, or `rz · x · rz` if middle angle ~0.
};

} // namespace mlir::qco::decomposition

// NOTE: We keep the small numeric helpers in this header because Euler
// decomposition/synthesis is the only current user and we want to avoid
// scattering tiny utilities across multiple files.
namespace mlir::qco::helpers {

/**
 * Wrap angle into interval [-pi, pi). If within atol of the endpoint, clamp to
 * -pi.
 */
[[nodiscard]] inline double mod2pi(double angle,
                                   double atol = mlir::utils::TOLERANCE) {
  // Wrap angle into the half-open interval [-pi, pi).
  // For non-finite values, keep the original (caller error / upstream issue).
  if (!std::isfinite(angle)) {
    return angle;
  }

  constexpr double pi = std::numbers::pi;
  constexpr double twoPi = 2.0 * std::numbers::pi;

  // Euclidean remainder of (angle + pi) modulo 2pi, then shift back by pi.
  // This ensures correct wrapping for negative angles as well.
  double r = std::fmod(angle + pi, twoPi);
  if (r < 0.0) {
    r += twoPi;
  }
  double wrapped = r - pi;

  // Canonicalize the upper endpoint back to -pi so callers always receive a
  // half-open interval [-pi, pi). We use an epsilon guard since rounding can
  // produce wrapped ~= +pi.
  if (wrapped >= pi - atol) {
    wrapped = -pi;
  }

  return wrapped;
}

} // namespace mlir::qco::helpers

namespace mlir::qco::decomposition {

/// Extract Euler parameters from single-qubit unitary matrices.
class EulerDecomposition {
public:
  /**
   * Extract canonical Euler parameters for `matrix` in the requested basis.
   *
   * Some target bases reuse the same parameter extraction routine and differ
   * only during circuit emission. The returned array always contains
   * `(theta, phi, lambda, phase)` in this order.
   */
  [[nodiscard]] static std::array<double, 4>
  anglesFromUnitary(const Eigen::Matrix2cd& matrix, EulerBasis basis);

private:
  /// Extract parameters for a `rz(phi) · ry(theta) · rz(lambda)` factorization.
  [[nodiscard]] static std::array<double, 4>
  paramsZYZ(const Eigen::Matrix2cd& matrix);

  /// Extract parameters for a `rz(phi) · rx(theta) · rz(lambda)` factorization.
  [[nodiscard]] static std::array<double, 4>
  paramsZXZ(const Eigen::Matrix2cd& matrix);

  /// Extract parameters for a `rx(phi) · ry(theta) · rx(lambda)` factorization.
  [[nodiscard]] static std::array<double, 4>
  paramsXYX(const Eigen::Matrix2cd& matrix);

  /// Extract parameters for a `rx(phi) · rz(theta) · rx(lambda)` factorization.
  [[nodiscard]] static std::array<double, 4>
  paramsXZX(const Eigen::Matrix2cd& matrix);

  /**
   * Extract Euler angles for `ZSXX` (`rz` / `sx`) synthesis.
   *
   * Reuses `(theta, phi, lambda)` from `paramsZYZ` and sets the scalar phase to
   * `phase - 0.5 * (theta + phi + lambda)` so `emitPSXGen` reproduces `matrix`
   * exactly, including the global phase induced by the `rz`/`sx`
   * parameterization.
   *
   * @note Adapted from `params_u1x_inner` in the IBM Qiskit framework.
   *       (C) Copyright IBM 2022
   *
   *       This code is licensed under the Apache License, Version 2.0. You may
   *       obtain a copy of this license in the LICENSE.txt file in the root
   *       directory of this source tree or at
   *       https://www.apache.org/licenses/LICENSE-2.0.
   *
   *       Any modifications or derivative works of this code must retain this
   *       copyright notice, and modified files need to carry a notice
   *       indicating that they have been altered from the originals.
   */
  [[nodiscard]] static std::array<double, 4>
  paramsPSX(const Eigen::Matrix2cd& matrix);
};

/// Parse a user-facing basis string (e.g. "zyz", "zsxx") into an Euler basis.
[[nodiscard]] std::optional<EulerBasis> parseEulerBasis(StringRef basis);

/// Emit an Euler-basis gate sequence implementing `targetMatrix` on `qubit`.
/// Returns the output qubit value.
///
/// The emitted circuit includes a `qco.gphase` correction when needed so the
/// overall unitary matches `targetMatrix` exactly (not only up to global
/// phase).
[[nodiscard]] Value
synthesizeUnitary1QEuler(OpBuilder& builder, Location loc, Value qubit,
                         const Eigen::Matrix2cd& targetMatrix,
                         EulerBasis basis);

} // namespace mlir::qco::decomposition
