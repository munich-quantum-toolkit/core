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

#include <gtest/gtest.h>
#include <llvm/ADT/StringRef.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>

TEST(GatePoweringTest, RecognizesFixedGatePowerPeriods) {
  for (llvm::StringRef gate : {"x", "y", "z", "h", "ecr", "rccx", "swap"}) {
    SCOPED_TRACE(gate.str());
    EXPECT_EQ(mlir::mqt::getFixedGatePowerPeriod(gate), 2U);
  }
  for (llvm::StringRef gate : {"s", "sdg", "sx", "sxdg", "iswap"}) {
    SCOPED_TRACE(gate.str());
    EXPECT_EQ(mlir::mqt::getFixedGatePowerPeriod(gate), 4U);
  }
  for (llvm::StringRef gate : {"t", "tdg"}) {
    SCOPED_TRACE(gate.str());
    EXPECT_EQ(mlir::mqt::getFixedGatePowerPeriod(gate), 8U);
  }
}

TEST(GatePoweringTest, DoesNotAssignFixedPeriodsToOtherGates) {
  for (llvm::StringRef gate : {"rx", "ry", "rz", "p", "u", "r", "dcx", ""}) {
    SCOPED_TRACE(gate.str());
    EXPECT_EQ(mlir::mqt::getFixedGatePowerPeriod(gate), 0U);
  }
}

TEST(GatePoweringTest, recognizesIntegerExponents) {
  EXPECT_TRUE(mlir::mqt::isIntegerExponent(-2.0));
  EXPECT_TRUE(mlir::mqt::isIntegerExponent(0.0));
  EXPECT_FALSE(mlir::mqt::isIntegerExponent(0.5));
  EXPECT_FALSE(
      mlir::mqt::isIntegerExponent(std::numeric_limits<double>::infinity()));
}

TEST(GatePoweringTest, recognizesEvenIntegerExponents) {
  EXPECT_TRUE(mlir::mqt::isEvenExponent(-2.0));
  EXPECT_TRUE(mlir::mqt::isEvenExponent(0.0));
  EXPECT_FALSE(mlir::mqt::isEvenExponent(3.0));
  EXPECT_FALSE(mlir::mqt::isEvenExponent(2.5));
  EXPECT_FALSE(
      mlir::mqt::isEvenExponent(std::numeric_limits<double>::infinity()));
}

using ReferenceComplex = std::complex<long double>;
using ReferenceMatrix = std::array<std::array<ReferenceComplex, 2>, 2>;

static ReferenceMatrix multiply(const ReferenceMatrix& lhs,
                                const ReferenceMatrix& rhs) {
  ReferenceMatrix product{};
  for (size_t row = 0; row < 2; ++row) {
    for (size_t column = 0; column < 2; ++column) {
      for (size_t inner = 0; inner < 2; ++inner) {
        product[row][column] += lhs[row][inner] * rhs[inner][column];
      }
    }
  }
  return product;
}

static ReferenceMatrix referenceUMatrix(double theta, double phi,
                                        double lambda) {
  const long double cosine = std::cos(static_cast<long double>(theta) / 2.0L);
  const long double sine = std::sin(static_cast<long double>(theta) / 2.0L);
  const ReferenceComplex phiPhase{std::cos(static_cast<long double>(phi)),
                                  std::sin(static_cast<long double>(phi))};
  const ReferenceComplex lambdaPhase{
      std::cos(static_cast<long double>(lambda)),
      std::sin(static_cast<long double>(lambda))};
  const ReferenceMatrix phasePhi{{{1.0L, 0.0L}, {0.0L, phiPhase}}};
  const ReferenceMatrix rotationY{{{cosine, -sine}, {sine, cosine}}};
  const ReferenceMatrix phaseLambda{{{1.0L, 0.0L}, {0.0L, lambdaPhase}}};
  // U(theta, phi, lambda) = P(phi) RY(theta) P(lambda).
  return multiply(multiply(phasePhi, rotationY), phaseLambda);
}

static void expectPowerPreservesMatrix(double theta, double phi, double lambda,
                                       unsigned exponent) {
  SCOPED_TRACE(testing::Message()
               << "theta=" << theta << ", phi=" << phi << ", lambda=" << lambda
               << ", exponent=" << exponent);
  const auto parameters =
      mlir::mqt::powerUParameters(theta, phi, lambda, exponent);
  ASSERT_TRUE(parameters);
  const auto source = referenceUMatrix(theta, phi, lambda);
  ReferenceMatrix expected{{{1.0L, 0.0L}, {0.0L, 1.0L}}};
  // Use sequential products rather than the implementation's binary powering.
  for (unsigned repetition = 0; repetition < exponent; ++repetition) {
    expected = multiply(source, expected);
  }

  const auto reconstructed = mlir::mqt::computeUMatrix(
      parameters->theta, parameters->phi, parameters->lambda);
  const auto globalPhase = std::polar(1.0, parameters->phase);
  for (size_t row = 0; row < 2; ++row) {
    for (size_t column = 0; column < 2; ++column) {
      SCOPED_TRACE(testing::Message()
                   << "matrix entry (" << row << ", " << column << ")");
      const ReferenceComplex actual =
          reconstructed[(2 * row) + column] * globalPhase;
      EXPECT_LE(std::abs(actual - expected[row][column]),
                mlir::mqt::U_POWER_EQUIVALENCE_TOLERANCE);
    }
  }
}

TEST(GatePoweringTest, PositiveIntegerPowersPreserveFullMatrix) {
  for (auto [theta, phi, lambda] : std::array{
           std::array{0.3, 0.4, -0.7},
           std::array{1.4, 2.2, -0.9},
           std::array{2.0 * std::numbers::pi, 0.2, -0.2},
       }) {
    /// General high powers may exceed the reconstruction bound.
    /// Diagonal and anti-diagonal cases below cover the maximum exponent.
    for (unsigned exponent : std::array{2U, 3U, 17U}) {
      ASSERT_NO_FATAL_FAILURE(
          expectPowerPreservesMatrix(theta, phi, lambda, exponent));
    }
  }
}

TEST(GatePoweringTest, DiagonalAndAntiDiagonalPowersPreserveGlobalPhase) {
  for (double theta : std::array{0.0, std::numbers::pi}) {
    for (unsigned exponent : std::array{2U, 3U, 17U, 1024U}) {
      ASSERT_NO_FATAL_FAILURE(
          expectPowerPreservesMatrix(theta, 0.3, 0.7, exponent));
    }
  }
}

TEST(GatePoweringTest, NearGimbalPowersPreserveFullMatrix) {
  for (double theta : std::array{
           1e-16,
           1e-12,
           std::numbers::pi - 1e-15,
           std::numbers::pi - 1e-12,
       }) {
    for (unsigned exponent : std::array{2U, 3U, 17U}) {
      ASSERT_NO_FATAL_FAILURE(
          expectPowerPreservesMatrix(theta, 0.31, 0.77, exponent));
    }
  }
}

TEST(GatePoweringTest, LargeEulerPhasePowersPreserveFullMatrix) {
  for (auto [phi, lambda] : std::array{
           std::array{1.0e16, 1.0},
           std::array{1.0, 1.0e16},
           std::array{1.0e16, -1.0e16},
           std::array{1.0e308, 1.0e308},
           std::array{-1.0e308, 1.0e308},
       }) {
    // Large powers can exceed the reconstruction bound within the exponent
    // limit. Stable cases above separately cover the maximum exponent.
    for (unsigned exponent : std::array{2U, 3U, 17U}) {
      ASSERT_NO_FATAL_FAILURE(
          expectPowerPreservesMatrix(0.3, phi, lambda, exponent));
    }
  }
}

TEST(GatePoweringTest, RejectsFinitePowerBeyondReconstructionBound) {
  /// This supported exponent still exceeds the full-matrix reconstruction
  /// bound.
  EXPECT_FALSE(mlir::mqt::powerUParameters(
      0.615926832310562, -2.7139721469341298, -2.7602783230969417, 1024.0));
}

TEST(GatePoweringTest, RejectsUnsupportedPowerExponents) {
  for (double exponent : std::array{
           -1.0,
           0.0,
           0.5,
           1.5,
           static_cast<double>(mlir::mqt::MAX_SAFE_U_POWER_EXPONENT) + 1.0,
           std::numeric_limits<double>::infinity(),
           -std::numeric_limits<double>::infinity(),
           std::numeric_limits<double>::quiet_NaN(),
       }) {
    SCOPED_TRACE(exponent);
    EXPECT_FALSE(mlir::mqt::powerUParameters(0.3, 0.4, 0.5, exponent));
  }
}

TEST(GatePoweringTest, RejectsNonfiniteInputAngles) {
  for (double invalid : std::array{
           std::numeric_limits<double>::infinity(),
           -std::numeric_limits<double>::infinity(),
           std::numeric_limits<double>::quiet_NaN(),
       }) {
    for (size_t index = 0; index < 3; ++index) {
      SCOPED_TRACE(testing::Message()
                   << "angle=" << index << ", value=" << invalid);
      std::array angles{0.3, 0.4, 0.5};
      angles[index] = invalid;
      EXPECT_FALSE(
          mlir::mqt::powerUParameters(angles[0], angles[1], angles[2], 2.0));
    }
  }
}
