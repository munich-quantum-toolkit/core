/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Utils/GatePowering.h"

#include "llvm/ADT/StringSwitch.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>

namespace mlir::mqt {

unsigned getFixedGatePowerPeriod(StringRef baseSymbol) {
  return llvm::StringSwitch<unsigned>(baseSymbol)
      .Cases({"x", "y", "z", "h", "ecr", "rccx", "swap"}, 2)
      .Cases({"s", "sdg", "sx", "sxdg", "iswap"}, 4)
      .Cases({"t", "tdg"}, 8)
      .Default(0);
}

std::array<std::complex<double>, 4> computeUMatrix(double theta, double phi,
                                                   double lambda) {
  using namespace std::complex_literals;
  const double cosine = std::cos(theta / 2.0);
  const double sine = std::sin(theta / 2.0);
  const auto phiPhase = std::exp(1i * phi);
  const auto lambdaPhase = std::exp(1i * lambda);
  return {
      cosine,
      -sine * lambdaPhase,
      sine * phiPhase,
      cosine * phiPhase * lambdaPhase,
  };
}

bool isIntegerExponent(const double value) {
  return value == std::floor(value) && std::isfinite(value);
}

bool isEvenExponent(const double value) {
  return isIntegerExponent(value) && std::fmod(std::fabs(value), 2.0) == 0.0;
}

std::optional<UPowerParameters>
powerUParameters(double theta, double phi, double lambda, double exponent) {
  if (!std::isfinite(theta) || !std::isfinite(phi) || !std::isfinite(lambda) ||
      !isIntegerExponent(exponent) || exponent <= 0.0 ||
      exponent > static_cast<double>(MAX_SAFE_U_POWER_EXPONENT)) {
    return std::nullopt;
  }

  using Matrix = std::array<std::complex<double>, 4>;
  const auto multiply = [](const Matrix& lhs, const Matrix& rhs) {
    return Matrix{
        (lhs[0] * rhs[0]) + (lhs[1] * rhs[2]),
        (lhs[0] * rhs[1]) + (lhs[1] * rhs[3]),
        (lhs[2] * rhs[0]) + (lhs[3] * rhs[2]),
        (lhs[2] * rhs[1]) + (lhs[3] * rhs[3]),
    };
  };
  Matrix base = computeUMatrix(theta, phi, lambda);
  Matrix sourcePower{1.0, 0.0, 0.0, 1.0};
  auto power = static_cast<uint64_t>(exponent);
  while (power != 0U) {
    if ((power & 1U) != 0U) {
      sourcePower = multiply(sourcePower, base);
    }
    power >>= 1U;
    if (power != 0U) {
      base = multiply(base, base);
    }
  }

  // Recover U's angles from the first column and the upper-right entry.
  // Diagonal and anti-diagonal matrices leave one phase unconstrained.
  const double cosine = std::abs(sourcePower[0]);
  const double sine = std::abs(sourcePower[2]);
  const double gimbalTolerance = 32.0 * std::numeric_limits<double>::epsilon();
  double resultTheta = 2.0 * std::atan2(sine, cosine);
  double resultPhase = std::arg(sourcePower[0]);
  double resultPhi = std::arg(sourcePower[2]) - resultPhase;
  double resultLambda = std::arg(-sourcePower[1]) - resultPhase;
  if (sine <= gimbalTolerance) {
    resultTheta = 0.0;
    resultPhi = 0.0;
    resultLambda = std::arg(sourcePower[3]) - resultPhase;
  } else if (cosine <= gimbalTolerance) {
    resultTheta = std::numbers::pi;
    resultPhase = 0.0;
    resultPhi = std::arg(sourcePower[2]);
    resultLambda = std::arg(-sourcePower[1]);
  }

  // Reject accumulated magnitude error and near-gimbal approximations that
  // exceed the full-matrix contract.
  Matrix reconstructed = computeUMatrix(resultTheta, resultPhi, resultLambda);
  const auto phase = std::polar(1.0, resultPhase);
  for (size_t i = 0; i < reconstructed.size(); ++i) {
    reconstructed[i] *= phase;
    // The negated comparison also rejects NaN.
    if (!(std::abs(sourcePower[i] - reconstructed[i]) <=
          U_POWER_EQUIVALENCE_TOLERANCE)) {
      return std::nullopt;
    }
  }
  return UPowerParameters{
      .theta = resultTheta,
      .phi = resultPhi,
      .lambda = resultLambda,
      .phase = resultPhase,
  };
}

} // namespace mlir::mqt
