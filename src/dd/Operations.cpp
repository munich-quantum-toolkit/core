/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Operations.hpp"

#include "dd/Complex.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Package.hpp"

#include <complex>
#include <stdexcept>
#include <vector>

namespace dd {

MatrixDD getGateDD(Package& dd, const GateType type,
                   const std::vector<fp>& params, const Controls& controls,
                   const Targets& targets) {
  if (isSingleQubitGate(type)) {
    if (targets.size() != 1) {
      throw std::invalid_argument(
          "Expected exactly one target qubit for single-qubit gate");
    }
    return dd.makeGateDD(opToSingleQubitGateMatrix(type, params), controls,
                         targets[0]);
  }
  if (isTwoQubitGate(type)) {
    if (targets.size() != 2) {
      throw std::invalid_argument(
          "Expected two target qubits for two-qubit gate");
    }
    return dd.makeTwoQubitGateDD(opToTwoQubitGateMatrix(type, params), controls,
                                 targets[0], targets[1]);
  }
  if (isThreeQubitGate(type)) {
    if (targets.size() != 3) {
      throw std::invalid_argument(
          "Expected three target qubits for three-qubit gate");
    }
    return dd.makeThreeQubitGateDD(opToThreeQubitGateMatrix(type, params),
                                   controls, targets[0], targets[1],
                                   targets[2]);
  }
  throw std::invalid_argument("Unsupported gate type");
}

VectorDD applyGlobalPhase(VectorDD& in, const fp& phase, Package& dd) {
  in.w = dd.cn.lookup(in.w * ComplexValue{std::polar(1.0, phase)});
  return in;
}

} // namespace dd
