/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/UnitaryMatrices.h"

#include "mlir/Dialect/QCO/Transforms/Decomposition/Gate.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateKind.h"

#include <Eigen/Core>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <unsupported/Eigen/KroneckerProduct>

namespace mlir::qco::decomposition {

Eigen::Matrix2cd uMatrix(double theta, double phi, double lambda) {
  return Eigen::Matrix2cd{{{{std::cos(theta / 2.), 0.},
                            {-std::cos(lambda) * std::sin(theta / 2.),
                             -std::sin(lambda) * std::sin(theta / 2.)}},
                           {{std::cos(phi) * std::sin(theta / 2.),
                             std::sin(phi) * std::sin(theta / 2.)},
                            {std::cos(lambda + phi) * std::cos(theta / 2.),
                             std::sin(lambda + phi) * std::cos(theta / 2.)}}}};
}

Eigen::Matrix2cd u2Matrix(double phi, double lambda) {
  return Eigen::Matrix2cd{
      {FRAC1_SQRT2,
       {-std::cos(lambda) * FRAC1_SQRT2, -std::sin(lambda) * FRAC1_SQRT2}},
      {{std::cos(phi) * FRAC1_SQRT2, std::sin(phi) * FRAC1_SQRT2},
       {std::cos(lambda + phi) * FRAC1_SQRT2,
        std::sin(lambda + phi) * FRAC1_SQRT2}}};
}

Eigen::Matrix2cd rxMatrix(double theta) {
  auto halfTheta = theta / 2.;
  auto cos = std::complex<double>{std::cos(halfTheta), 0.};
  auto isin = std::complex<double>{0., -std::sin(halfTheta)};
  return Eigen::Matrix2cd{{cos, isin}, {isin, cos}};
}

Eigen::Matrix2cd ryMatrix(double theta) {
  auto halfTheta = theta / 2.;
  std::complex<double> cos{std::cos(halfTheta), 0.};
  std::complex<double> sin{std::sin(halfTheta), 0.};
  return Eigen::Matrix2cd{{cos, -sin}, {sin, cos}};
}

Eigen::Matrix2cd rzMatrix(double theta) {
  return Eigen::Matrix2cd{{{std::cos(theta / 2.), -std::sin(theta / 2.)}, 0},
                          {0, {std::cos(theta / 2.), std::sin(theta / 2.)}}};
}

Eigen::Matrix4cd rxxMatrix(double theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return Eigen::Matrix4cd{{cosTheta, 0, 0, {0., -sinTheta}},
                          {0, cosTheta, {0., -sinTheta}, 0},
                          {0, {0., -sinTheta}, cosTheta, 0},
                          {{0., -sinTheta}, 0, 0, cosTheta}};
}

Eigen::Matrix4cd ryyMatrix(double theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return Eigen::Matrix4cd{{{cosTheta, 0, 0, {0., sinTheta}},
                           {0, cosTheta, {0., -sinTheta}, 0},
                           {0, {0., -sinTheta}, cosTheta, 0},
                           {{0., sinTheta}, 0, 0, cosTheta}}};
}

Eigen::Matrix4cd rzzMatrix(double theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return Eigen::Matrix4cd{{{cosTheta, -sinTheta}, 0, 0, 0},
                          {0, {cosTheta, sinTheta}, 0, 0},
                          {0, 0, {cosTheta, sinTheta}, 0},
                          {0, 0, 0, {cosTheta, -sinTheta}}};
}

Eigen::Matrix2cd pMatrix(double lambda) {
  return Eigen::Matrix2cd{{1, 0}, {0, {std::cos(lambda), std::sin(lambda)}}};
}

Eigen::Matrix4cd expandToTwoQubits(const Eigen::Matrix2cd& singleQubitMatrix,
                                   QubitId qubitId) {
  if (qubitId == 0) {
    return Eigen::kroneckerProduct(singleQubitMatrix,
                                   Eigen::Matrix2cd::Identity());
  }
  if (qubitId == 1) {
    return Eigen::kroneckerProduct(Eigen::Matrix2cd::Identity(),
                                   singleQubitMatrix);
  }
  llvm::reportFatalInternalError("Invalid qubit id for single-qubit expansion");
}

Eigen::Matrix2cd getSingleQubitMatrix(const Gate& gate) {
  if (gate.type == GateKind::SX) {
    return Eigen::Matrix2cd{
        {std::complex<double>{0.5, 0.5}, std::complex<double>{0.5, -0.5}},
        {std::complex<double>{0.5, -0.5}, std::complex<double>{0.5, 0.5}}};
  }
  if (gate.type == GateKind::RX) {
    assert(gate.parameter.size() == 1);
    return rxMatrix(gate.parameter[0]);
  }
  if (gate.type == GateKind::RY) {
    assert(gate.parameter.size() == 1);
    return ryMatrix(gate.parameter[0]);
  }
  if (gate.type == GateKind::RZ) {
    assert(gate.parameter.size() == 1);
    return rzMatrix(gate.parameter[0]);
  }
  if (gate.type == GateKind::X) {
    return Eigen::Matrix2cd{{0, 1}, {1, 0}};
  }
  if (gate.type == GateKind::I) {
    return Eigen::Matrix2cd::Identity();
  }
  if (gate.type == GateKind::P) {
    assert(gate.parameter.size() == 1);
    return pMatrix(gate.parameter[0]);
  }
  if (gate.type == GateKind::U) {
    assert(gate.parameter.size() == 3);
    const double theta = gate.parameter[0];
    const double phi = gate.parameter[1];
    const double lambda = gate.parameter[2];
    return uMatrix(theta, phi, lambda);
  }
  if (gate.type == GateKind::U2) {
    assert(gate.parameter.size() == 2);
    const double phi = gate.parameter[0];
    const double lambda = gate.parameter[1];
    return u2Matrix(phi, lambda);
  }
  if (gate.type == GateKind::H) {
    return H_GATE;
  }
  llvm::reportFatalInternalError(
      "unsupported gate type for single qubit matrix");
}

// Reconstruct a two-qubit workspace matrix for a decomposition `Gate`.
// Used by sequence verification and `QubitGateSequence::getUnitaryMatrix()`.
Eigen::Matrix4cd getTwoQubitMatrix(const Gate& gate) {
  if (gate.qubitId.empty()) {
    if (gate.type == GateKind::I) {
      return Eigen::Matrix4cd::Identity();
    }
    llvm::reportFatalInternalError(
        "Invalid gate: empty qubit IDs are only allowed for identity");
  }
  if (gate.qubitId.size() == 1) {
    return expandToTwoQubits(getSingleQubitMatrix(gate), gate.qubitId[0]);
  }
  if (gate.qubitId.size() == 2) {
    const bool validPair01 =
        gate.qubitId == llvm::SmallVector<QubitId, 2>{0, 1};
    const bool validPair10 =
        gate.qubitId == llvm::SmallVector<QubitId, 2>{1, 0};
    if (!validPair01 && !validPair10) {
      llvm::reportFatalInternalError(
          "Invalid two-qubit gate qubit IDs: expected {0,1} or {1,0}");
    }
    if (gate.type == GateKind::X) {
      // Controlled-X (`cx`) is directional: swapping `{control, target}`
      // changes the operator. We therefore handle both orderings explicitly.
      //
      // The two matrices below represent `cx` for the two possible
      // `Gate::qubitId` orderings. Qubit 0 is the MSB of the 4x4 computational
      // basis (matching `UnitaryOpInterface::getUnitaryMatrix4x4`), so
      // `{0,1}` and `{1,0}` lead to different basis-layout matrices.
      if (validPair01) {
        // control = wire 0 (MSB), target = wire 1.
        return Eigen::Matrix4cd{
            {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
      }
      if (validPair10) {
        // control = wire 1, target = wire 0 (MSB).
        return Eigen::Matrix4cd{
            {1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}};
      }
      llvm::reportFatalInternalError("Invalid qubit IDs for CX gate");
    }
    if (gate.type == GateKind::Z) {
      // Controlled-Z (`cz`) is symmetric in its two qubits; swapping `{0,1}`
      // and
      // `{1,0}` yields the same operator, so one matrix is sufficient.
      return Eigen::Matrix4cd{
          {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, -1}};
    }
    if (gate.type == GateKind::RXX) {
      assert(gate.parameter.size() == 1);
      return rxxMatrix(gate.parameter[0]);
    }
    if (gate.type == GateKind::RYY) {
      assert(gate.parameter.size() == 1);
      return ryyMatrix(gate.parameter[0]);
    }
    if (gate.type == GateKind::RZZ) {
      assert(gate.parameter.size() == 1);
      return rzzMatrix(gate.parameter[0]);
    }
    if (gate.type == GateKind::I) {
      return Eigen::Matrix4cd::Identity();
    }
    llvm::reportFatalInternalError(
        "Unsupported gate type for two qubit matrix");
  }
  llvm::reportFatalInternalError(
      "Invalid number of qubit IDs for two-qubit matrix construction");
}

} // namespace mlir::qco::decomposition
