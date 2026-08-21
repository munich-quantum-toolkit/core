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

#include <cstdint>
#include <optional>

namespace mlir::mqt {

/**
 * Maximum exponent considered for safe binary64 U-gate powering.
 *
 * Even with analytical SU(2) powering, uncertainty in the input angles is
 * magnified by the exponent. Candidate rewrites are additionally checked
 * against the source matrix before they are accepted.
 */
inline constexpr uint64_t MAX_SAFE_U_POWER_EXPONENT = 1024U;

/// Maximum entry-wise matrix error accepted for a powered U-gate rewrite.
inline constexpr double U_POWER_EQUIVALENCE_TOLERANCE = 5e-13;

/**
 * @brief Parameters representing a powered U gate.
 *
 * All values are in radians. The phase satisfies
 * `U(input)^exponent = exp(i * phase) * U(theta, phi, lambda)`.
 */
struct UPowerParameters {
  double theta;  ///< Resulting U rotation angle.
  double phi;    ///< Resulting U phi angle.
  double lambda; ///< Resulting U lambda angle.
  double phase;  ///< Remaining global phase.
};

/// Check whether a floating-point exponent is an integer.
[[nodiscard]] bool isIntegerExponent(double value);

/// Check whether a floating-point exponent is an even integer.
[[nodiscard]] bool isEvenExponent(double value);

/**
 * @brief Compute a positive integral power of a constant U gate.
 *
 * @return Parameters satisfying
 * `U(theta, phi, lambda)^exponent = exp(i*phase) * U(result)`, or
 * `std::nullopt` if @p exponent is not a positive integer no greater than
 * `MAX_SAFE_U_POWER_EXPONENT`, an input is not finite, or the binary64 result
 * cannot be reconstructed within `U_POWER_EQUIVALENCE_TOLERANCE`.
 */
[[nodiscard]] std::optional<UPowerParameters>
powerUParameters(double theta, double phi, double lambda, double exponent);

} // namespace mlir::mqt
