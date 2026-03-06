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

#include "EulerBasis.h"
#include "GateSequence.h"
#include "ir/operations/OpType.hpp"

#include <Eigen/Core>
#include <array>
#include <optional>

namespace mlir::qco::decomposition {

/**
 * Decomposition of single-qubit matrices into rotation gates using a KAK
 * decomposition.
 *
 * @note only the following euler bases are supported for now:
 *       ZYZ, ZXZ, XYX and XZX
 */
class EulerDecomposition {
public:
  /**
   * Perform single-qubit decomposition of a 2x2 unitary matrix based on a
   * given euler basis.
   */
  [[nodiscard]] static OneQubitGateSequence
  generateCircuit(EulerBasis targetBasis, const Eigen::Matrix2cd& unitaryMatrix,
                  bool simplify, std::optional<double> atol);

  /**
   * Calculate angles of a single-qubit matrix according to the given
   * EulerBasis.
   *
   * @return array containing (theta, phi, lambda, phase) in this order
   */
  static std::array<double, 4> anglesFromUnitary(const Eigen::Matrix2cd& matrix,
                                                 EulerBasis basis);

private:
  static std::array<double, 4> paramsZyz(const Eigen::Matrix2cd& matrix);

  static std::array<double, 4> paramsZxz(const Eigen::Matrix2cd& matrix);

  static std::array<double, 4> paramsXyx(const Eigen::Matrix2cd& matrix);

  static std::array<double, 4> paramsXzx(const Eigen::Matrix2cd& matrix);

  /**
   * @note Adapted from circuit_kak() in the IBM Qiskit framework.
   *       (C) Copyright IBM 2022
   *
   *       This code is licensed under the Apache License, Version 2.0. You may
   *       obtain a copy of this license in the LICENSE.txt file in the root
   *       directory of this source tree or at
   *       http://www.apache.org/licenses/LICENSE-2.0.
   *
   *       Any modifications or derivative works of this code must retain this
   *       copyright notice, and modified files need to carry a notice
   *       indicating that they have been altered from the originals.
   */
  [[nodiscard]] static OneQubitGateSequence
  decomposeKAK(double theta, double phi, double lambda, double phase,
               qc::OpType kGate, qc::OpType aGate, bool simplify,
               std::optional<double> atol);
};
} // namespace mlir::qco::decomposition
