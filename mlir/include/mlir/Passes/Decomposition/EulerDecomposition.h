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
#include "Helpers.h"
#include "ir/Definitions.hpp"
#include "ir/operations/OpType.hpp"

#include <Eigen/Core>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <llvm/ADT/STLExtras.h>
#include <optional>
#include <stdexcept>

namespace mlir::qco::decomposition {

/**
 * Decomposition of single-qubit matrices into rotation gates using a KAK
 * decomposition.
 */
class EulerDecomposition {
public:
  /**
   * Perform single-qubit decomposition of a 2x2 unitary matrix based on a
   * given euler basis.
   */
  [[nodiscard]] static OneQubitGateSequence
  generateCircuit(EulerBasis targetBasis, const matrix2x2& unitaryMatrix,
                  bool simplify, std::optional<fp> atol) {
    auto [theta, phi, lambda, phase] =
        anglesFromUnitary(unitaryMatrix, targetBasis);

    switch (targetBasis) {
    case EulerBasis::ZYZ:
      return decomposeKAK(theta, phi, lambda, phase, qc::RZ, qc::RY, simplify,
                          atol);
    case EulerBasis::ZXZ:
      return decomposeKAK(theta, phi, lambda, phase, qc::RZ, qc::RX, simplify,
                          atol);
    case EulerBasis::XZX:
      return decomposeKAK(theta, phi, lambda, phase, qc::RX, qc::RZ, simplify,
                          atol);
    case EulerBasis::XYX:
      return decomposeKAK(theta, phi, lambda, phase, qc::RX, qc::RY, simplify,
                          atol);
    default:
      throw std::invalid_argument{"Unsupported base for circuit generation!"};
    }
  }

  /**
   * Calculate angles of a single-qubit matrix according to the given
   * EulerBasis.
   *
   * @return array containing (theta, phi, lambda, phase) in this order
   */
  static std::array<fp, 4> anglesFromUnitary(const matrix2x2& matrix,
                                             EulerBasis basis) {
    if (basis == EulerBasis::XYX) {
      return paramsXyxInner(matrix);
    }
    if (basis == EulerBasis::XZX) {
      return paramsXzxInner(matrix);
    }
    if (basis == EulerBasis::ZYZ) {
      return paramsZyzInner(matrix);
    }
    if (basis == EulerBasis::ZXZ) {
      return paramsZxzInner(matrix);
    }
    throw std::invalid_argument{"Unknown EulerBasis for angles_from_unitary"};
  }

private:
  static std::array<fp, 4> paramsZyzInner(const matrix2x2& matrix) {
    const auto detArg = std::arg(matrix.determinant());
    const auto phase = 0.5 * detArg;
    const auto theta =
        2. * std::atan2(std::abs(matrix(1, 0)), std::abs(matrix(0, 0)));
    const auto ang1 = std::arg(matrix(1, 1));
    const auto ang2 = std::arg(matrix(1, 0));
    const auto phi = ang1 + ang2 - detArg;
    const auto lam = ang1 - ang2;
    return {theta, phi, lam, phase};
  }

  static std::array<fp, 4> paramsZxzInner(const matrix2x2& matrix) {
    const auto [theta, phi, lam, phase] = paramsZyzInner(matrix);
    return {theta, phi + (qc::PI / 2.), lam - (qc::PI / 2.), phase};
  }

  static std::array<fp, 4> paramsXyxInner(const matrix2x2& matrix) {
    const matrix2x2 matZyz{
        {static_cast<fp>(0.5) *
             (matrix(0, 0) + matrix(0, 1) + matrix(1, 0) + matrix(1, 1)),
         static_cast<fp>(0.5) *
             (matrix(0, 0) - matrix(0, 1) + matrix(1, 0) - matrix(1, 1))},
        {static_cast<fp>(0.5) *
             (matrix(0, 0) + matrix(0, 1) - matrix(1, 0) - matrix(1, 1)),
         static_cast<fp>(0.5) *
             (matrix(0, 0) - matrix(0, 1) - matrix(1, 0) + matrix(1, 1))},
    };
    auto [theta, phi, lam, phase] = paramsZyzInner(matZyz);
    auto newPhi = helpers::mod2pi(phi + qc::PI, 0.);
    auto newLam = helpers::mod2pi(lam + qc::PI, 0.);
    return {
        theta,
        newPhi,
        newLam,
        phase + ((newPhi + newLam - phi - lam) / 2.),
    };
  }

  static std::array<fp, 4> paramsXzxInner(const matrix2x2& matrix) {
    auto det = matrix.determinant();
    auto phase = std::imag(std::log(det)) / 2.0;
    auto sqrtDet = std::sqrt(det);
    const matrix2x2 matZyz{
        {
            {(matrix(0, 0) / sqrtDet).real(), (matrix(1, 0) / sqrtDet).imag()},
            {(matrix(1, 0) / sqrtDet).real(), (matrix(0, 0) / sqrtDet).imag()},
        },
        {
            {-(matrix(1, 0) / sqrtDet).real(), (matrix(0, 0) / sqrtDet).imag()},
            {(matrix(0, 0) / sqrtDet).real(), -(matrix(1, 0) / sqrtDet).imag()},
        },
    };
    auto [theta, phi, lam, phase_zxz] = paramsZxzInner(matZyz);
    return {theta, phi, lam, phase + phase_zxz};
  }

  /**
   * @note Adapted from circuit_kak() in the IBM Qiskit framework.
   *       (C) Copyright IBM 2022
   *
   *       This code is licensed under the Apache License, Version 2.0. You
   * may obtain a copy of this license in the LICENSE.txt file in the root
   *       directory of this source tree or at
   *       http://www.apache.org/licenses/LICENSE-2.0.
   *
   *       Any modifications or derivative works of this code must retain
   * this copyright notice, and modified files need to carry a notice
   *       indicating that they have been altered from the originals.
   */
  [[nodiscard]] static OneQubitGateSequence
  decomposeKAK(fp theta, fp phi, fp lambda, fp phase, qc::OpType kGate,
               qc::OpType aGate, bool simplify, std::optional<fp> atol) {
    fp angleZeroEpsilon = atol.value_or(DEFAULT_ATOL);
    if (!simplify) {
      // setting atol to negative value to make all angle checks true; this will
      // effectively disable the simplification since all rotations appear to be
      // "necessary"
      angleZeroEpsilon = -1.0;
    }

    OneQubitGateSequence sequence{
        .gates = {},
        .globalPhase = phase - ((phi + lambda) / 2.),
    };
    if (std::abs(theta) <= angleZeroEpsilon) {
      lambda += phi;
      lambda = helpers::mod2pi(lambda);
      if (std::abs(lambda) > angleZeroEpsilon) {
        sequence.gates.push_back({.type = kGate, .parameter = {lambda}});
        sequence.globalPhase += lambda / 2.0;
      }
      return sequence;
    }

    if (std::abs(theta - qc::PI) <= angleZeroEpsilon) {
      sequence.globalPhase += phi;
      lambda -= phi;
      phi = 0.0;
    }
    if (std::abs(helpers::mod2pi(lambda + qc::PI)) <= angleZeroEpsilon ||
        std::abs(helpers::mod2pi(phi + qc::PI)) <= angleZeroEpsilon) {
      lambda += qc::PI;
      theta = -theta;
      phi += qc::PI;
    }
    lambda = helpers::mod2pi(lambda);
    if (std::abs(lambda) > angleZeroEpsilon) {
      sequence.globalPhase += lambda / 2.0;
      sequence.gates.push_back({.type = kGate, .parameter = {lambda}});
    }
    sequence.gates.push_back({.type = aGate, .parameter = {theta}});
    phi = helpers::mod2pi(phi);
    if (std::abs(phi) > angleZeroEpsilon) {
      sequence.globalPhase += phi / 2.0;
      sequence.gates.push_back({.type = kGate, .parameter = {phi}});
    }
    return sequence;
  }
};
} // namespace mlir::qco::decomposition
