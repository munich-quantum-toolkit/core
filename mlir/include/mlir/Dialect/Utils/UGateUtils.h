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

#include "mlir/Dialect/Utils/Utils.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <optional>

namespace mlir::utils {

/**
 * Maximum exponent for numerically stable binary64 U-gate powering.
 *
 * The 2^10 cap keeps the O(n * epsilon) error accumulated by repeated
 * squaring below the 1e-12 tolerance used by full-matrix equivalence tests.
 */
inline constexpr uint64_t MAX_SAFE_U_POWER_EXPONENT = 1024U;

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

/**
 * @brief Compute a positive integral power of a constant U gate.
 *
 * @return Parameters satisfying
 * `U(theta, phi, lambda)^exponent = exp(i*phase) * U(result)`, or
 * `std::nullopt` if @p exponent is not a positive integer no greater than
 * `MAX_SAFE_U_POWER_EXPONENT`.
 */
[[nodiscard]] inline std::optional<UPowerParameters>
powerUParameters(const double theta, const double phi, const double lambda,
                 const double exponent) {
  // Repeated squaring magnifies binary64 roundoff with every squared power.
  if (!mlir::utils::isIntegerExponent(exponent) || exponent <= 0.0 ||
      exponent > static_cast<double>(MAX_SAFE_U_POWER_EXPONENT)) {
    return std::nullopt;
  }

  using Complex = std::complex<double>;
  using Matrix = std::array<Complex, 4>;
  const auto multiply = [](const Matrix& lhs, const Matrix& rhs) {
    return Matrix{(lhs[0] * rhs[0]) + (lhs[1] * rhs[2]),
                  (lhs[0] * rhs[1]) + (lhs[1] * rhs[3]),
                  (lhs[2] * rhs[0]) + (lhs[3] * rhs[2]),
                  (lhs[2] * rhs[1]) + (lhs[3] * rhs[3])};
  };

  const double halfTheta = theta / 2.0;
  const double c = std::cos(halfTheta);
  const double s = std::sin(halfTheta);
  const Complex imaginary{0.0, 1.0};
  Matrix base{c, -s * std::exp(imaginary * lambda),
              s * std::exp(imaginary * phi),
              c * std::exp(imaginary * (phi + lambda))};
  Matrix result{1.0, 0.0, 0.0, 1.0};
  auto power = static_cast<uint64_t>(exponent);
  while (power != 0U) {
    if ((power & 1U) != 0U) {
      result = multiply(result, base);
    }
    power >>= 1U;
    if (power != 0U) {
      base = multiply(base, base);
    }
  }

  const Complex determinant = (result[0] * result[3]) - (result[1] * result[2]);
  const double determinantArgument = std::arg(determinant);
  const double resultTheta =
      2.0 * std::atan2(std::abs(result[2]), std::abs(result[0]));
  const double angle1 = std::arg(result[3]);
  const double angle2 =
      std::abs(result[2]) > TOLERANCE ? std::arg(result[2]) : 0.0;
  const double resultPhi = angle1 + angle2 - determinantArgument;
  const double resultLambda = angle1 - angle2;
  const double resultPhase =
      0.5 * (determinantArgument - resultPhi - resultLambda);
  return UPowerParameters{resultTheta, resultPhi, resultLambda, resultPhase};
}

} // namespace mlir::utils
