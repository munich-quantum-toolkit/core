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
#include "Helpers.h"
#include "ir/operations/OpType.hpp"

#include <unsupported/Eigen/KroneckerProduct>

namespace mlir::qco::decomposition {

inline constexpr double SQRT2 = 1.414213562373095048801688724209698079L;
inline constexpr double FRAC1_SQRT2 =
    0.707106781186547524400844362104849039284835937688474036588L;

[[nodiscard]] constexpr Eigen::Matrix2cd
uMatrix(const double lambda, const double phi, const double theta) {
  return Eigen::Matrix2cd{{{{std::cos(theta / 2.), 0.},
                            {-std::cos(lambda) * std::sin(theta / 2.),
                             -std::sin(lambda) * std::sin(theta / 2.)}},
                           {{std::cos(phi) * std::sin(theta / 2.),
                             std::sin(phi) * std::sin(theta / 2.)},
                            {std::cos(lambda + phi) * std::cos(theta / 2.),
                             std::sin(lambda + phi) * std::cos(theta / 2.)}}}};
}

[[nodiscard]] constexpr Eigen::Matrix2cd u2Matrix(const double lambda,
                                                  const double phi) {
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
  auto cos = std::complex<double>{std::cos(halfTheta), 0.};
  auto sin = std::complex<double>{std::sin(halfTheta), 0.};
  return Eigen::Matrix2cd{{cos, -sin}, {sin, cos}};
}

[[nodiscard]] constexpr Eigen::Matrix2cd rzMatrix(double theta) {
  return Eigen::Matrix2cd{
      {std::complex<double>{std::cos(theta / 2.), -std::sin(theta / 2.)}, 0},
      {0, std::complex<double>{std::cos(theta / 2.), std::sin(theta / 2.)}}};
}

[[nodiscard]] constexpr Eigen::Matrix4cd rxxMatrix(const double theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return Eigen::Matrix4cd{{cosTheta, C_ZERO, C_ZERO, {0., -sinTheta}},
                          {C_ZERO, cosTheta, {0., -sinTheta}, C_ZERO},
                          {C_ZERO, {0., -sinTheta}, cosTheta, C_ZERO},
                          {{0., -sinTheta}, C_ZERO, C_ZERO, cosTheta}};
}

[[nodiscard]] constexpr Eigen::Matrix4cd ryyMatrix(const double theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return Eigen::Matrix4cd{{{cosTheta, 0, 0, {0., sinTheta}},
                           {0, cosTheta, {0., -sinTheta}, 0},
                           {0, {0., -sinTheta}, cosTheta, 0},
                           {{0., sinTheta}, 0, 0, cosTheta}}};
}

[[nodiscard]] constexpr Eigen::Matrix4cd rzzMatrix(const double theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return Eigen::Matrix4cd{
      {std::complex<double>{cosTheta, -sinTheta}, C_ZERO, C_ZERO, C_ZERO},
      {C_ZERO, {cosTheta, sinTheta}, C_ZERO, C_ZERO},
      {C_ZERO, C_ZERO, {cosTheta, sinTheta}, C_ZERO},
      {C_ZERO, C_ZERO, C_ZERO, {cosTheta, -sinTheta}}};
}

[[nodiscard]] constexpr Eigen::Matrix2cd pMatrix(const double lambda) {
  return Eigen::Matrix2cd{{1, 0}, {0, {std::cos(lambda), std::sin(lambda)}}};
}
constexpr Eigen::Matrix4cd SWAP_GATE{
    {1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}};
constexpr Eigen::Matrix2cd H_GATE{{1.0 / SQRT2, 1.0 / SQRT2},
                                  {1.0 / SQRT2, -1.0 / SQRT2}};
constexpr Eigen::Matrix2cd IPZ{{C_IM, C_ZERO}, {C_ZERO, C_M_IM}};
constexpr Eigen::Matrix2cd IPY{{C_ZERO, C_ONE}, {C_M_ONE, C_ZERO}};
constexpr Eigen::Matrix2cd IPX{{C_ZERO, C_IM}, {C_IM, C_ZERO}};

[[nodiscard]] inline Eigen::Matrix4cd
expandToTwoQubits(const Eigen::Matrix2cd& singleQubitMatrix, QubitId qubitId) {
  if (qubitId == 0) {
    return Eigen::kroneckerProduct(Eigen::Matrix2cd::Identity(),
                                   singleQubitMatrix);
  }
  if (qubitId == 1) {
    return Eigen::kroneckerProduct(singleQubitMatrix,
                                   Eigen::Matrix2cd::Identity());
  }
  throw std::invalid_argument{"Invalid qubit id for single-qubit expansion"};
}

[[nodiscard]] inline Eigen::Matrix4cd
fixTwoQubitMatrixQubitOrder(const Eigen::Matrix4cd& twoQubitMatrix,
                            const llvm::SmallVector<QubitId, 2>& qubitIds) {
  if (qubitIds == llvm::SmallVector<QubitId, 2>{1, 0}) {
    // since UnitaryOpInterface::getUnitaryMatrix() does have a static
    // qubit order, adjust if we need the other direction of the gate
    return decomposition::SWAP_GATE * twoQubitMatrix * decomposition::SWAP_GATE;
  }
  if (qubitIds == llvm::SmallVector<QubitId, 2>{0, 1}) {
    return twoQubitMatrix;
  }
  throw std::invalid_argument{"Invalid qubit IDs for fixing two-qubit matrix"};
}

[[nodiscard]] inline Eigen::Matrix2cd getSingleQubitMatrix(const Gate& gate) {
  if (gate.type == qc::SX) {
    return Eigen::Matrix2cd{
        {std::complex<double>{0.5, 0.5}, std::complex<double>{0.5, -0.5}},
        {std::complex<double>{0.5, -0.5}, std::complex<double>{0.5, 0.5}}};
  }
  if (gate.type == qc::RX) {
    return rxMatrix(gate.parameter[0]);
  }
  if (gate.type == qc::RY) {
    return ryMatrix(gate.parameter[0]);
  }
  if (gate.type == qc::RZ) {
    return rzMatrix(gate.parameter[0]);
  }
  if (gate.type == qc::X) {
    return Eigen::Matrix2cd{{0, 1}, {1, 0}};
  }
  if (gate.type == qc::I) {
    return Eigen::Matrix2cd::Identity();
  }
  if (gate.type == qc::P) {
    return pMatrix(gate.parameter[0]);
  }
  if (gate.type == qc::U) {
    return uMatrix(gate.parameter[0], gate.parameter[1], gate.parameter[2]);
  }
  if (gate.type == qc::U2) {
    return u2Matrix(gate.parameter[0], gate.parameter[1]);
  }
  if (gate.type == qc::H) {
    return H_GATE;
  }
  throw std::invalid_argument{
      "unsupported gate type for single qubit matrix (" +
      qc::toString(gate.type) + ")"};
}

// TODO: remove? only used for verification of circuit and in unittests
[[nodiscard]] inline Eigen::Matrix4cd getTwoQubitMatrix(const Gate& gate) {
  if (gate.qubitId.empty()) {
    return Eigen::Matrix4cd::Identity();
  }
  if (gate.qubitId.size() == 1) {
    return expandToTwoQubits(getSingleQubitMatrix(gate), gate.qubitId[0]);
  }
  if (gate.qubitId.size() == 2) {
    if (gate.type == qc::X) {
      // controlled X (CX)
      if (gate.qubitId == llvm::SmallVector<QubitId, 2>{0, 1}) {
        return Eigen::Matrix4cd{
            {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
      }
      if (gate.qubitId == llvm::SmallVector<QubitId, 2>{1, 0}) {
        return Eigen::Matrix4cd{
            {1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}};
      }
    }
    if (gate.type == qc::RXX) {
      return rxxMatrix(gate.parameter[0]);
    }
    if (gate.type == qc::RYY) {
      return ryyMatrix(gate.parameter[0]);
    }
    if (gate.type == qc::RZZ) {
      return rzzMatrix(gate.parameter[0]);
    }
    if (gate.type == qc::I) {
      return Eigen::Matrix4cd::Identity();
    }
    throw std::invalid_argument{"unsupported gate type for two qubit matrix (" +
                                qc::toString(gate.type) + ")"};
  }
  throw std::logic_error{"Invalid number of qubit IDs in compute_unitary"};
}

} // namespace mlir::qco::decomposition
