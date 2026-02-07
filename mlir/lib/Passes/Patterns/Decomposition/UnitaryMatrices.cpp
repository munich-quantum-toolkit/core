/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Passes/Decomposition/UnitaryMatrices.h"

#include "ir/operations/OpType.hpp"
#include "mlir/Passes/Decomposition/Gate.h"

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <complex>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <unsupported/Eigen/KroneckerProduct>

namespace mlir::qco::decomposition {

Eigen::Matrix2cd uMatrix(double lambda, double phi, double theta) {
  return Eigen::Matrix2cd{{{{std::cos(theta / 2.), 0.},
                            {-std::cos(lambda) * std::sin(theta / 2.),
                             -std::sin(lambda) * std::sin(theta / 2.)}},
                           {{std::cos(phi) * std::sin(theta / 2.),
                             std::sin(phi) * std::sin(theta / 2.)},
                            {std::cos(lambda + phi) * std::cos(theta / 2.),
                             std::sin(lambda + phi) * std::cos(theta / 2.)}}}};
}

Eigen::Matrix2cd u2Matrix(double lambda, double phi) {
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
    return Eigen::kroneckerProduct(Eigen::Matrix2cd::Identity(),
                                   singleQubitMatrix);
  }
  if (qubitId == 1) {
    return Eigen::kroneckerProduct(singleQubitMatrix,
                                   Eigen::Matrix2cd::Identity());
  }
  llvm::reportFatalInternalError("Invalid qubit id for single-qubit expansion");
}

Eigen::Matrix4cd
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
  llvm::reportFatalInternalError(
      "Invalid qubit IDs for fixing two-qubit matrix");
}

Eigen::Matrix2cd getSingleQubitMatrix(const Gate& gate) {
  if (gate.type == qc::SX) {
    return Eigen::Matrix2cd{
        {std::complex<double>{0.5, 0.5}, std::complex<double>{0.5, -0.5}},
        {std::complex<double>{0.5, -0.5}, std::complex<double>{0.5, 0.5}}};
  }
  if (gate.type == qc::RX) {
    assert(gate.parameter.size() == 1);
    return rxMatrix(gate.parameter[0]);
  }
  if (gate.type == qc::RY) {
    assert(gate.parameter.size() == 1);
    return ryMatrix(gate.parameter[0]);
  }
  if (gate.type == qc::RZ) {
    assert(gate.parameter.size() == 1);
    return rzMatrix(gate.parameter[0]);
  }
  if (gate.type == qc::X) {
    return Eigen::Matrix2cd{{0, 1}, {1, 0}};
  }
  if (gate.type == qc::I) {
    return Eigen::Matrix2cd::Identity();
  }
  if (gate.type == qc::P) {
    assert(gate.parameter.size() == 1);
    return pMatrix(gate.parameter[0]);
  }
  if (gate.type == qc::U) {
    assert(gate.parameter.size() == 3);
    return uMatrix(gate.parameter[0], gate.parameter[1], gate.parameter[2]);
  }
  if (gate.type == qc::U2) {
    assert(gate.parameter.size() == 2);
    return u2Matrix(gate.parameter[0], gate.parameter[1]);
  }
  if (gate.type == qc::H) {
    return H_GATE;
  }
  llvm::reportFatalInternalError(
      llvm::StringRef("unsupported gate type for single qubit matrix (" +
                      qc::toString(gate.type) + ")"));
}

// TODO: remove? only used for verification of circuit and in unittests
Eigen::Matrix4cd getTwoQubitMatrix(const Gate& gate) {
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
      llvm::reportFatalInternalError("Invalid qubit IDs for CX gate");
    }
    if (gate.type == qc::RXX) {
      assert(gate.parameter.size() == 1);
      return rxxMatrix(gate.parameter[0]);
    }
    if (gate.type == qc::RYY) {
      assert(gate.parameter.size() == 1);
      return ryyMatrix(gate.parameter[0]);
    }
    if (gate.type == qc::RZZ) {
      assert(gate.parameter.size() == 1);
      return rzzMatrix(gate.parameter[0]);
    }
    if (gate.type == qc::I) {
      return Eigen::Matrix4cd::Identity();
    }
    llvm::reportFatalInternalError(
        llvm::StringRef("Unsupported gate type for two qubit matrix (" +
                        qc::toString(gate.type) + ")"));
  }
  llvm::reportFatalInternalError(
      "Invalid number of qubit IDs in compute_unitary");
}

} // namespace mlir::qco::decomposition
