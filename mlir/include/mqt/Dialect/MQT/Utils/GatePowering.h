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

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"

#include <array>
#include <complex>
#include <cstdint>
#include <optional>

namespace mlir::mqt {

/// Return the exact exponent period of a supported fixed named gate, or zero
/// for a gate without such a period.
///
/// All returned periods are powers of two, so reducing a finite binary64
/// exponent does not lose a fractional part.
[[nodiscard]] unsigned getFixedGatePowerPeriod(StringRef baseSymbol);

/// Evaluate U(θ, φ, λ) in row-major order for finite input angles. Compose
/// phase factors without adding angles, so large finite parameters cannot
/// overflow or absorb a fixed phase offset. Callers validate finiteness.
[[nodiscard]] std::array<std::complex<double>, 4>
computeUMatrix(double theta, double phi, double lambda);

/// Maximum exponent considered for safe binary64 U-gate powering.
///
/// Repeated matrix multiplication accumulates rounding error. Candidate
/// rewrites are checked against the powered source matrix before acceptance.
inline constexpr uint64_t MAX_SAFE_U_POWER_EXPONENT = 1024U;

/// Maximum entry-wise matrix error accepted for a powered U-gate rewrite.
inline constexpr double U_POWER_EQUIVALENCE_TOLERANCE = 5e-13;

/// Parameters representing a powered U gate.
///
/// All values are in radians. The phase satisfies
/// `U(input)^exponent = exp(i * phase) * U(θ, φ, λ)`.
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

/// Compute a positive integral power of a constant U gate.
///
/// @return Parameters satisfying
/// `U(θ, φ, λ)^exponent = exp(i*phase) * U(result)`, or
/// `std::nullopt` if @p exponent is not a positive integer no greater than
/// `MAX_SAFE_U_POWER_EXPONENT`, an input is not finite, or the binary64 result
/// cannot be reconstructed within `U_POWER_EQUIVALENCE_TOLERANCE`.
[[nodiscard]] std::optional<UPowerParameters>
powerUParameters(double theta, double phi, double lambda, double exponent);

} // namespace mlir::mqt
