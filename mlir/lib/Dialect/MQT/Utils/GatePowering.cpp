/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/Utils/GatePowering.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>

namespace mlir::mqt {

bool isIntegerExponent(const double value) {
  return value == std::floor(value) && std::isfinite(value);
}

bool isEvenExponent(const double value) {
  return isIntegerExponent(value) && std::fmod(std::fabs(value), 2.0) == 0.0;
}

std::optional<UPowerParameters> powerUParameters(const double theta,
                                                 const double phi,
                                                 const double lambda,
                                                 const double exponent) {
  if (!std::isfinite(theta) || !std::isfinite(phi) || !std::isfinite(lambda) ||
      !isIntegerExponent(exponent) || exponent <= 0.0 ||
      exponent > static_cast<double>(MAX_SAFE_U_POWER_EXPONENT)) {
    return std::nullopt;
  }

  using Quaternion = std::array<double, 4>;
  using Complex = std::complex<double>;
  using Matrix = std::array<Complex, 4>;

  /// U(theta, phi, lambda) = exp(i * (phi + lambda) / 2) *
  /// RZ(phi) * RY(theta) * RZ(lambda). Represent the SU(2) factor by a unit
  /// quaternion and power it analytically by multiplying its axis angle.
  const double halfTheta = theta / 2.0;
  const double halfPhi = phi / 2.0;
  const double halfLambda = lambda / 2.0;
  const double cosTheta = std::cos(halfTheta);
  const double sinTheta = std::sin(halfTheta);
  const double cosPhi = std::cos(halfPhi);
  const double sinPhi = std::sin(halfPhi);
  const double cosLambda = std::cos(halfLambda);
  const double sinLambda = std::sin(halfLambda);
  Quaternion quaternion{
      cosTheta * ((cosPhi * cosLambda) - (sinPhi * sinLambda)),
      sinTheta * ((cosPhi * sinLambda) - (sinPhi * cosLambda)),
      sinTheta * ((cosPhi * cosLambda) + (sinPhi * sinLambda)),
      cosTheta * ((cosPhi * sinLambda) + (sinPhi * cosLambda))};
  const double quaternionNorm =
      std::hypot(std::hypot(quaternion[0], quaternion[1]),
                 std::hypot(quaternion[2], quaternion[3]));
  if (!std::isfinite(quaternionNorm) || quaternionNorm == 0.0) {
    return std::nullopt;
  }
  for (double& component : quaternion) {
    component /= quaternionNorm;
  }

  const auto integralExponent = static_cast<uint64_t>(exponent);
  const double vectorNorm =
      std::hypot(std::hypot(quaternion[1], quaternion[2]), quaternion[3]);
  Quaternion poweredQuaternion{};
  if (vectorNorm == 0.0) {
    poweredQuaternion[0] =
        quaternion[0] < 0.0 && (integralExponent & 1U) != 0U ? -1.0 : 1.0;
  } else {
    const double axisAngle = std::atan2(vectorNorm, quaternion[0]);
    const double poweredAxisAngle =
        std::remainder(exponent * axisAngle, 2.0 * std::numbers::pi);
    const double vectorScale = std::sin(poweredAxisAngle) / vectorNorm;
    poweredQuaternion = {
        std::cos(poweredAxisAngle), quaternion[1] * vectorScale,
        quaternion[2] * vectorScale, quaternion[3] * vectorScale};
  }
  const double poweredNorm =
      std::hypot(std::hypot(poweredQuaternion[0], poweredQuaternion[1]),
                 std::hypot(poweredQuaternion[2], poweredQuaternion[3]));
  if (!std::isfinite(poweredNorm) || poweredNorm == 0.0) {
    return std::nullopt;
  }
  for (double& component : poweredQuaternion) {
    component /= poweredNorm;
  }

  const auto [w, x, y, z] = poweredQuaternion;
  const double transverseNorm = std::hypot(x, y);
  const double axialNorm = std::hypot(w, z);
  const double gimbalTolerance = 32.0 * std::numeric_limits<double>::epsilon();
  double resultTheta = 2.0 * std::atan2(transverseNorm, axialNorm);
  double resultPhi = 0.0;
  double resultLambda = 0.0;
  if (transverseNorm <= gimbalTolerance) {
    resultTheta = 0.0;
    resultPhi = 2.0 * std::atan2(z, w);
  } else if (axialNorm <= gimbalTolerance) {
    resultTheta = std::numbers::pi;
    resultPhi = 2.0 * std::atan2(-x, y);
  } else {
    const double angleSum = std::atan2(z, w);
    const double angleDifference = std::atan2(-x, y);
    resultPhi = angleSum + angleDifference;
    resultLambda = angleSum - angleDifference;
  }

  constexpr double fourPi = 4.0 * std::numbers::pi;
  const double inputPhase =
      std::remainder((std::remainder(phi, fourPi) / 2.0) +
                         (std::remainder(lambda, fourPi) / 2.0),
                     2.0 * std::numbers::pi);
  const double poweredPhase =
      std::remainder(exponent * inputPhase, 2.0 * std::numbers::pi);
  const double resultPhase =
      std::remainder(poweredPhase - ((resultPhi + resultLambda) / 2.0),
                     2.0 * std::numbers::pi);

  /// Binary64 evaluation of the source U matrix can itself deviate from an
  /// exact unitary for large angles, and powering magnifies that deviation.
  /// Reject a rewrite when the analytical unitary cannot represent the source
  /// operation closely enough for the dialect's full-matrix contract.
  const Complex imaginary{0.0, 1.0};
  const auto uMatrix = [&](const double matrixTheta, const double matrixPhi,
                           const double matrixLambda) {
    const double matrixCos = std::cos(matrixTheta / 2.0);
    const double matrixSin = std::sin(matrixTheta / 2.0);
    return Matrix{matrixCos,
                  matrixSin *
                      std::exp(imaginary * (matrixLambda + std::numbers::pi)),
                  matrixSin * std::exp(imaginary * matrixPhi),
                  matrixCos * std::exp(imaginary * (matrixPhi + matrixLambda))};
  };
  const auto multiply = [](const Matrix& lhs, const Matrix& rhs) {
    return Matrix{(lhs[0] * rhs[0]) + (lhs[1] * rhs[2]),
                  (lhs[0] * rhs[1]) + (lhs[1] * rhs[3]),
                  (lhs[2] * rhs[0]) + (lhs[3] * rhs[2]),
                  (lhs[2] * rhs[1]) + (lhs[3] * rhs[3])};
  };
  Matrix base = uMatrix(theta, phi, lambda);
  Matrix sourcePower{1.0, 0.0, 0.0, 1.0};
  auto power = integralExponent;
  while (power != 0U) {
    if ((power & 1U) != 0U) {
      sourcePower = multiply(sourcePower, base);
    }
    power >>= 1U;
    if (power != 0U) {
      base = multiply(base, base);
    }
  }
  Matrix reconstructed = uMatrix(resultTheta, resultPhi, resultLambda);
  const Complex phase = std::exp(imaginary * resultPhase);
  for (size_t i = 0; i < reconstructed.size(); ++i) {
    reconstructed[i] *= phase;
    if (!std::isfinite(sourcePower[i].real()) ||
        !std::isfinite(sourcePower[i].imag()) ||
        std::abs(sourcePower[i] - reconstructed[i]) >
            U_POWER_EQUIVALENCE_TOLERANCE) {
      return std::nullopt;
    }
  }
  return UPowerParameters{.theta = resultTheta,
                          .phi = resultPhi,
                          .lambda = resultLambda,
                          .phase = resultPhase};
}

} // namespace mlir::mqt
