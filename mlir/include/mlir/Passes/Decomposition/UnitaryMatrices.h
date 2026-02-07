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

#include "Gate.h"

#include <Eigen/Core>
#include <cmath>
#include <complex>

namespace mlir::qco::decomposition {

inline constexpr double SQRT2 = 1.414213562373095048801688724209698079L;
inline constexpr double FRAC1_SQRT2 =
    0.707106781186547524400844362104849039284835937688474036588L;

[[nodiscard]] constexpr Eigen::Matrix2cd uMatrix(double lambda, double phi,
                                                 double theta) {
  return Eigen::Matrix2cd{{{{std::cos(theta / 2.), 0.},
                            {-std::cos(lambda) * std::sin(theta / 2.),
                             -std::sin(lambda) * std::sin(theta / 2.)}},
                           {{std::cos(phi) * std::sin(theta / 2.),
                             std::sin(phi) * std::sin(theta / 2.)},
                            {std::cos(lambda + phi) * std::cos(theta / 2.),
                             std::sin(lambda + phi) * std::cos(theta / 2.)}}}};
}

[[nodiscard]] constexpr Eigen::Matrix2cd u2Matrix(double lambda, double phi) {
  return Eigen::Matrix2cd{
      {FRAC1_SQRT2,
       {-std::cos(lambda) * FRAC1_SQRT2, -std::sin(lambda) * FRAC1_SQRT2}},
      {{std::cos(phi) * FRAC1_SQRT2, std::sin(phi) * FRAC1_SQRT2},
       {std::cos(lambda + phi) * FRAC1_SQRT2,
        std::sin(lambda + phi) * FRAC1_SQRT2}}};
}

[[nodiscard]] constexpr Eigen::Matrix2cd rxMatrix(double theta) {
  auto halfTheta = theta / 2.;
  auto cos = std::complex<double>{std::cos(halfTheta), 0.};
  auto isin = std::complex<double>{0., -std::sin(halfTheta)};
  return Eigen::Matrix2cd{{cos, isin}, {isin, cos}};
}

[[nodiscard]] constexpr Eigen::Matrix2cd ryMatrix(double theta) {
  auto halfTheta = theta / 2.;
  std::complex<double> cos{std::cos(halfTheta), 0.};
  std::complex<double> sin{std::sin(halfTheta), 0.};
  return Eigen::Matrix2cd{{cos, -sin}, {sin, cos}};
}

[[nodiscard]] constexpr Eigen::Matrix2cd rzMatrix(double theta) {
  return Eigen::Matrix2cd{{{std::cos(theta / 2.), -std::sin(theta / 2.)}, 0},
                          {0, {std::cos(theta / 2.), std::sin(theta / 2.)}}};
}

[[nodiscard]] constexpr Eigen::Matrix4cd rxxMatrix(double theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return Eigen::Matrix4cd{{cosTheta, 0, 0, {0., -sinTheta}},
                          {0, cosTheta, {0., -sinTheta}, 0},
                          {0, {0., -sinTheta}, cosTheta, 0},
                          {{0., -sinTheta}, 0, 0, cosTheta}};
}

[[nodiscard]] constexpr Eigen::Matrix4cd ryyMatrix(double theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return Eigen::Matrix4cd{{{cosTheta, 0, 0, {0., sinTheta}},
                           {0, cosTheta, {0., -sinTheta}, 0},
                           {0, {0., -sinTheta}, cosTheta, 0},
                           {{0., sinTheta}, 0, 0, cosTheta}}};
}

[[nodiscard]] constexpr Eigen::Matrix4cd rzzMatrix(double theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return Eigen::Matrix4cd{{{cosTheta, -sinTheta}, 0, 0, 0},
                          {0, {cosTheta, sinTheta}, 0, 0},
                          {0, 0, {cosTheta, sinTheta}, 0},
                          {0, 0, 0, {cosTheta, -sinTheta}}};
}

[[nodiscard]] constexpr Eigen::Matrix2cd pMatrix(double lambda) {
  return Eigen::Matrix2cd{{1, 0}, {0, {std::cos(lambda), std::sin(lambda)}}};
}

inline const Eigen::Matrix4cd SWAP_GATE{
    {1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}};
inline const Eigen::Matrix2cd H_GATE{{FRAC1_SQRT2, FRAC1_SQRT2},
                                     {FRAC1_SQRT2, -FRAC1_SQRT2}};
inline const Eigen::Matrix2cd IPZ{{{0, 1}, 0}, {0, {0, -1}}};
inline const Eigen::Matrix2cd IPY{{0, 1}, {-1, 0}};
inline const Eigen::Matrix2cd IPX{{0, {0, 1}}, {{0, 1}, 0}};

[[nodiscard]] Eigen::Matrix4cd
expandToTwoQubits(const Eigen::Matrix2cd& singleQubitMatrix, QubitId qubitId);

[[nodiscard]] Eigen::Matrix4cd
fixTwoQubitMatrixQubitOrder(const Eigen::Matrix4cd& twoQubitMatrix,
                            const llvm::SmallVector<QubitId, 2>& qubitIds);

[[nodiscard]] Eigen::Matrix2cd getSingleQubitMatrix(const Gate& gate);

// TODO: remove? only used for verification of circuit and in unittests
[[nodiscard]] Eigen::Matrix4cd getTwoQubitMatrix(const Gate& gate);

} // namespace mlir::qco::decomposition
