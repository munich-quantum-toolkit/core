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

namespace mlir::qco::decomposition {

inline constexpr auto SQRT2 =
    static_cast<fp>(1.414213562373095048801688724209698079L);
inline constexpr auto FRAC1_SQRT2 = static_cast<fp>(
    0.707106781186547524400844362104849039284835937688474036588L);

[[nodiscard]] inline matrix2x2 uMatrix(const fp lambda, const fp phi,
                                       const fp theta) {
  return matrix2x2{{{{std::cos(theta / 2.), 0.},
                     {-std::cos(lambda) * std::sin(theta / 2.),
                      -std::sin(lambda) * std::sin(theta / 2.)}},
                    {{std::cos(phi) * std::sin(theta / 2.),
                      std::sin(phi) * std::sin(theta / 2.)},
                     {std::cos(lambda + phi) * std::cos(theta / 2.),
                      std::sin(lambda + phi) * std::cos(theta / 2.)}}}};
}

[[nodiscard]] inline matrix2x2 u2Matrix(const fp lambda, const fp phi) {
  return matrix2x2{
      {FRAC1_SQRT2,
       {-std::cos(lambda) * FRAC1_SQRT2, -std::sin(lambda) * FRAC1_SQRT2}},
      {{std::cos(phi) * FRAC1_SQRT2, std::sin(phi) * FRAC1_SQRT2},
       {std::cos(lambda + phi) * FRAC1_SQRT2,
        std::sin(lambda + phi) * FRAC1_SQRT2}}};
}

inline matrix2x2 rxMatrix(fp theta) {
  auto halfTheta = theta / 2.;
  auto cos = qfp(std::cos(halfTheta), 0.);
  auto isin = qfp(0., -std::sin(halfTheta));
  return matrix2x2{{cos, isin}, {isin, cos}};
}

inline matrix2x2 ryMatrix(fp theta) {
  auto halfTheta = theta / 2.;
  auto cos = qfp(std::cos(halfTheta), 0.);
  auto sin = qfp(std::sin(halfTheta), 0.);
  return matrix2x2{{cos, -sin}, {sin, cos}};
}

inline matrix2x2 rzMatrix(fp theta) {
  return matrix2x2{{qfp{std::cos(theta / 2.), -std::sin(theta / 2.)}, 0},
                   {0, qfp{std::cos(theta / 2.), std::sin(theta / 2.)}}};
}

inline matrix4x4 rxxMatrix(const fp theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return matrix4x4{{cosTheta, C_ZERO, C_ZERO, {0., -sinTheta}},
                   {C_ZERO, cosTheta, {0., -sinTheta}, C_ZERO},
                   {C_ZERO, {0., -sinTheta}, cosTheta, C_ZERO},
                   {{0., -sinTheta}, C_ZERO, C_ZERO, cosTheta}};
}

inline matrix4x4 ryyMatrix(const fp theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return matrix4x4{{{cosTheta, 0, 0, {0., sinTheta}},
                    {0, cosTheta, {0., -sinTheta}, 0},
                    {0, {0., -sinTheta}, cosTheta, 0},
                    {{0., sinTheta}, 0, 0, cosTheta}}};
}

inline matrix4x4 rzzMatrix(const fp theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return matrix4x4{{qfp{cosTheta, -sinTheta}, C_ZERO, C_ZERO, C_ZERO},
                   {C_ZERO, {cosTheta, sinTheta}, C_ZERO, C_ZERO},
                   {C_ZERO, C_ZERO, {cosTheta, sinTheta}, C_ZERO},
                   {C_ZERO, C_ZERO, C_ZERO, {cosTheta, -sinTheta}}};
}

inline matrix2x2 pMatrix(const fp lambda) {
  return matrix2x2{{1, 0}, {0, {std::cos(lambda), std::sin(lambda)}}};
}
const matrix2x2 IDENTITY_GATE = matrix2x2::Identity();
const matrix2x2 H_GATE{{1.0 / SQRT2, 1.0 / SQRT2}, {1.0 / SQRT2, -1.0 / SQRT2}};
const matrix2x2 IPZ{{IM, C_ZERO}, {C_ZERO, M_IM}};
const matrix2x2 IPY{{C_ZERO, C_ONE}, {C_M_ONE, C_ZERO}};
const matrix2x2 IPX{{C_ZERO, IM}, {IM, C_ZERO}};

inline matrix2x2 getSingleQubitMatrix(const Gate& gate) {
  if (gate.type == qc::SX) {
    return matrix2x2{{qfp{0.5, 0.5}, qfp{0.5, -0.5}},
                     {qfp{0.5, -0.5}, qfp{0.5, 0.5}}};
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
    return matrix2x2{{0, 1}, {1, 0}};
  }
  if (gate.type == qc::I) {
    return IDENTITY_GATE;
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
    return matrix2x2{{FRAC1_SQRT2, FRAC1_SQRT2}, {FRAC1_SQRT2, -FRAC1_SQRT2}};
  }
  throw std::invalid_argument{
      "unsupported gate type for single qubit matrix (" +
      qc::toString(gate.type) + ")"};
}

inline matrix4x4 getTwoQubitMatrix(const Gate& gate) {
  using helpers::kroneckerProduct;

  if (gate.qubitId.empty()) {
    return kroneckerProduct(IDENTITY_GATE, IDENTITY_GATE);
  }
  if (gate.qubitId.size() == 1) {
    if (gate.qubitId[0] == 0) {
      return kroneckerProduct(IDENTITY_GATE, getSingleQubitMatrix(gate));
    }
    if (gate.qubitId[0] == 1) {
      return kroneckerProduct(getSingleQubitMatrix(gate), IDENTITY_GATE);
    }
    throw std::logic_error{"Invalid qubit ID in getTwoQubitMatrix"};
  }
  if (gate.qubitId.size() == 2) {
    if (gate.type == qc::X) {
      // controlled X (CX)
      if (gate.qubitId == llvm::SmallVector<QubitId, 2>{0, 1}) {
        return matrix4x4{
            {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
      }
      if (gate.qubitId == llvm::SmallVector<QubitId, 2>{1, 0}) {
        return matrix4x4{
            {1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}};
      }
    }
    if (gate.type == qc::RXX) {
      // TODO: check qubit order?
      return rxxMatrix(gate.parameter[0]);
    }
    if (gate.type == qc::RYY) {
      // TODO: check qubit order?
      return ryyMatrix(gate.parameter[0]);
    }
    if (gate.type == qc::RZZ) {
      // TODO: check qubit order?
      return rzzMatrix(gate.parameter[0]);
    }
    if (gate.type == qc::I) {
      return kroneckerProduct(IDENTITY_GATE, IDENTITY_GATE);
    }
    throw std::invalid_argument{"unsupported gate type for two qubit matrix (" +
                                qc::toString(gate.type) + ")"};
  }
  throw std::logic_error{"Invalid number of qubit IDs in compute_unitary"};
}

} // namespace mlir::qco::decomposition
