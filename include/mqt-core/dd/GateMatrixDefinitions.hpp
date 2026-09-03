/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file GateMatrixDefinitions.hpp
 * @brief Gate-matrix definitions used to construct decision diagrams.
 */

#pragma once

#include "dd/DDDefinitions.hpp"

#include <cstdint>
#include <vector>

namespace dd {

/// Gates supported by the DD package.
enum class GateType : std::uint8_t {
  None,
  I,
  H,
  X,
  Y,
  Z,
  S,
  Sdg,
  T,
  Tdg,
  U,
  U2,
  P,
  SX,
  SXdg,
  RX,
  RY,
  RZ,
  R,
  SWAP,
  iSWAP,
  DCX,
  ECR,
  RXX,
  RYY,
  RZZ,
  RZX,
  XXminusYY,
  XXplusYY,
  RCCX
};

[[nodiscard]] constexpr bool isSingleQubitGate(const GateType type) {
  switch (type) {
  case GateType::I:
  case GateType::H:
  case GateType::X:
  case GateType::Y:
  case GateType::Z:
  case GateType::S:
  case GateType::Sdg:
  case GateType::T:
  case GateType::Tdg:
  case GateType::U:
  case GateType::U2:
  case GateType::P:
  case GateType::SX:
  case GateType::SXdg:
  case GateType::RX:
  case GateType::RY:
  case GateType::RZ:
  case GateType::R:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] constexpr bool isTwoQubitGate(const GateType type) {
  switch (type) {
  case GateType::SWAP:
  case GateType::iSWAP:
  case GateType::DCX:
  case GateType::ECR:
  case GateType::RXX:
  case GateType::RYY:
  case GateType::RZZ:
  case GateType::RZX:
  case GateType::XXminusYY:
  case GateType::XXplusYY:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] constexpr bool isThreeQubitGate(const GateType type) {
  return type == GateType::RCCX;
}

/// Single-qubit gate matrix for collapsing a qubit to the |0> state
constexpr GateMatrix MEAS_ZERO_MAT{1, 0, 0, 0};
/// Single-qubit gate matrix for collapsing a qubit to the |1> state
constexpr GateMatrix MEAS_ONE_MAT{0, 0, 0, 1};

/**
 * @brief Converts a given quantum operation to a single-qubit gate matrix
 * @param t The quantum operation to convert
 * @param params The parameters of the quantum operation
 * @return The single-qubit gate matrix representation of the quantum operation
 */
GateMatrix opToSingleQubitGateMatrix(GateType t,
                                     const std::vector<fp>& params = {});

/**
 * @brief Converts a given quantum operation to a two-qubit gate matrix
 * @param t The quantum operation to convert
 * @param params The parameters of the quantum operation
 * @return The two-qubit gate matrix representation of the quantum operation
 */
TwoQubitGateMatrix opToTwoQubitGateMatrix(GateType t,
                                          const std::vector<fp>& params = {});

/**
 * @brief Converts a given quantum operation to a three-qubit gate matrix
 * @param t The quantum operation to convert
 * @param params The parameters of the quantum operation
 * @return The three-qubit gate matrix representation of the quantum operation
 */
ThreeQubitGateMatrix
opToThreeQubitGateMatrix(GateType t, const std::vector<fp>& params = {});

} // namespace dd
