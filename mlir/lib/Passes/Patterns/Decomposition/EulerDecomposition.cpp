/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Passes/Decomposition/EulerDecomposition.h"

#include "ir/Definitions.hpp"
#include "ir/operations/OpType.hpp"
#include "mlir/Passes/Decomposition/EulerBasis.h"
#include "mlir/Passes/Decomposition/GateSequence.h"
#include "mlir/Passes/Decomposition/Helpers.h"

#include <Eigen/Core>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <optional>

namespace mlir::qco::decomposition {

OneQubitGateSequence
EulerDecomposition::generateCircuit(EulerBasis targetBasis,
                                    const Eigen::Matrix2cd& unitaryMatrix,
                                    bool simplify, std::optional<double> atol) {
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
    llvm::reportFatalInternalError("Unsupported euler basis for circuit "
                                   "generation in decomposition!");
  }
}

std::array<double, 4>
EulerDecomposition::anglesFromUnitary(const Eigen::Matrix2cd& matrix,
                                      EulerBasis basis) {
  if (basis == EulerBasis::XYX) {
    return paramsXyx(matrix);
  }
  if (basis == EulerBasis::XZX) {
    return paramsXzx(matrix);
  }
  if (basis == EulerBasis::ZYZ) {
    return paramsZyz(matrix);
  }
  if (basis == EulerBasis::ZXZ) {
    return paramsZxz(matrix);
  }
  llvm::reportFatalInternalError(
      "Unsupported euler basis for angle computation in decomposition!");
}

std::array<double, 4>
EulerDecomposition::paramsZyz(const Eigen::Matrix2cd& matrix) {
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

std::array<double, 4>
EulerDecomposition::paramsZxz(const Eigen::Matrix2cd& matrix) {
  const auto [theta, phi, lam, phase] = paramsZyz(matrix);
  return {theta, phi + (qc::PI / 2.), lam - (qc::PI / 2.), phase};
}

std::array<double, 4>
EulerDecomposition::paramsXyx(const Eigen::Matrix2cd& matrix) {
  const Eigen::Matrix2cd matZyz{
      {0.5 * (matrix(0, 0) + matrix(0, 1) + matrix(1, 0) + matrix(1, 1)),
       0.5 * (matrix(0, 0) - matrix(0, 1) + matrix(1, 0) - matrix(1, 1))},
      {0.5 * (matrix(0, 0) + matrix(0, 1) - matrix(1, 0) - matrix(1, 1)),
       0.5 * (matrix(0, 0) - matrix(0, 1) - matrix(1, 0) + matrix(1, 1))},
  };
  auto [theta, phi, lam, phase] = paramsZyz(matZyz);
  auto newPhi = helpers::mod2pi(phi + qc::PI, 0.);
  auto newLam = helpers::mod2pi(lam + qc::PI, 0.);
  return {
      theta,
      newPhi,
      newLam,
      phase + ((newPhi + newLam - phi - lam) / 2.),
  };
}

std::array<double, 4>
EulerDecomposition::paramsXzx(const Eigen::Matrix2cd& matrix) {
  auto det = matrix.determinant();
  auto phase = std::imag(std::log(det)) / 2.0;
  auto sqrtDet = std::sqrt(det);
  const Eigen::Matrix2cd matZxz{
      {
          {(matrix(0, 0) / sqrtDet).real(), (matrix(1, 0) / sqrtDet).imag()},
          {(matrix(1, 0) / sqrtDet).real(), (matrix(0, 0) / sqrtDet).imag()},
      },
      {
          {-(matrix(1, 0) / sqrtDet).real(), (matrix(0, 0) / sqrtDet).imag()},
          {(matrix(0, 0) / sqrtDet).real(), -(matrix(1, 0) / sqrtDet).imag()},
      },
  };
  auto [theta, phi, lam, phase_zxz] = paramsZxz(matZxz);
  return {theta, phi, lam, phase + phase_zxz};
}

OneQubitGateSequence EulerDecomposition::decomposeKAK(
    double theta, double phi, double lambda, double phase, qc::OpType kGate,
    qc::OpType aGate, bool simplify, std::optional<double> atol) {
  double angleZeroEpsilon = atol.value_or(DEFAULT_ATOL);
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

} // namespace mlir::qco::decomposition
