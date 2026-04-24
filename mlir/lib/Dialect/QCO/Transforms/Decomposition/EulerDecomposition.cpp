/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/EulerDecomposition.h"

#include "mlir/Dialect/QCO/Transforms/Decomposition/EulerBasis.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateKind.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateSequence.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Helpers.h"

#include <Eigen/Core>
#include <llvm/Support/ErrorHandling.h>

#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <optional>

namespace mlir::qco::decomposition {

OneQubitGateSequence
EulerDecomposition::generateCircuit(EulerBasis targetBasis,
                                    const Eigen::Matrix2cd& unitaryMatrix,
                                    bool simplify, std::optional<double> atol) {
  // First normalize the input into basis-specific Euler parameters, then map
  // those parameters to the target gate alphabet.
  auto [theta, phi, lambda, phase] =
      anglesFromUnitary(unitaryMatrix, targetBasis);

  switch (targetBasis) {
  case EulerBasis::ZYZ:
    return decomposeKAK(theta, phi, lambda, phase, GateKind::RZ, GateKind::RY,
                        simplify, atol);
  case EulerBasis::ZXZ:
    return decomposeKAK(theta, phi, lambda, phase, GateKind::RZ, GateKind::RX,
                        simplify, atol);
  case EulerBasis::XZX:
    return decomposeKAK(theta, phi, lambda, phase, GateKind::RX, GateKind::RZ,
                        simplify, atol);
  case EulerBasis::XYX:
    return decomposeKAK(theta, phi, lambda, phase, GateKind::RX, GateKind::RY,
                        simplify, atol);
  case EulerBasis::U:
    [[fallthrough]];
  case EulerBasis::U3:
    [[fallthrough]];
  case EulerBasis::U321:
    return OneQubitGateSequence{
        .gates = {{.type = GateKind::U, .parameter = {theta, phi, lambda}}},
        .globalPhase = phase - ((phi + lambda) / 2.),
    };
  case EulerBasis::ZSX:
    return decomposePsxGen(theta, phi, lambda, phase, /*allowXShortcut=*/false,
                           simplify, atol);
  case EulerBasis::ZSXX:
    return decomposePsxGen(theta, phi, lambda, phase, /*allowXShortcut=*/true,
                           simplify, atol);
  }
  llvm::reportFatalInternalError(
      "Unsupported euler basis for circuit generation in decomposition!");
}

std::array<double, 4>
EulerDecomposition::anglesFromUnitary(const Eigen::Matrix2cd& matrix,
                                      EulerBasis basis) {
  switch (basis) {
  case EulerBasis::XYX:
    return paramsXyx(matrix);
  case EulerBasis::XZX:
    return paramsXzx(matrix);
  case EulerBasis::ZYZ:
    return paramsZyz(matrix);
  case EulerBasis::ZXZ:
    return paramsZxz(matrix);
  case EulerBasis::U:
  case EulerBasis::U3:
  case EulerBasis::U321:
    // The `u` gate parameterization is derived from the standard Z-Y-Z form.
    return paramsZyz(matrix);
  case EulerBasis::ZSX:
  case EulerBasis::ZSXX:
    // Qiskit's `params_u1x_inner` reuses Z-Y-Z angles but shifts the global
    // phase by `-0.5 * (theta + phi + lambda)` so that the decomposition
    // matches an `rz`/`sx` emission exactly (not only up to global phase).
    return paramsU1x(matrix);
  }
  llvm::reportFatalInternalError(
      "Unsupported euler basis for angle computation in decomposition!");
}

std::array<double, 4>
EulerDecomposition::paramsZyz(const Eigen::Matrix2cd& matrix) {
  // Split the matrix determinant into a scalar phase and an SU(2) part, then
  // recover the canonical Z-Y-Z angles from the relative entry magnitudes and
  // phases.
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
  // Convert from the Z-Y-Z parameterization via the standard basis-change
  // identity RX(a) = RZ(pi/2) RY(a) RZ(-pi/2).
  const auto [theta, phi, lam, phase] = paramsZyz(matrix);
  return {theta, phi + (std::numbers::pi / 2.0), lam - (std::numbers::pi / 2.0),
          phase};
}

std::array<double, 4>
EulerDecomposition::paramsXyx(const Eigen::Matrix2cd& matrix) {
  // Conjugating by Hadamards transforms an X-Y-X decomposition problem into a
  // Z-Y-Z one, so we solve it there and map the angles back.
  const Eigen::Matrix2cd matZyz{
      {0.5 * (matrix(0, 0) + matrix(0, 1) + matrix(1, 0) + matrix(1, 1)),
       0.5 * (matrix(0, 0) - matrix(0, 1) + matrix(1, 0) - matrix(1, 1))},
      {0.5 * (matrix(0, 0) + matrix(0, 1) - matrix(1, 0) - matrix(1, 1)),
       0.5 * (matrix(0, 0) - matrix(0, 1) - matrix(1, 0) + matrix(1, 1))},
  };
  auto [theta, phi, lam, phase] = paramsZyz(matZyz);
  auto newPhi = helpers::mod2pi(phi + std::numbers::pi, 0.);
  auto newLam = helpers::mod2pi(lam + std::numbers::pi, 0.);
  return {
      theta,
      newPhi,
      newLam,
      phase + ((newPhi + newLam - phi - lam) / 2.),
  };
}

std::array<double, 4>
EulerDecomposition::paramsU1x(const Eigen::Matrix2cd& matrix) {
  // The determinant of the rz/sx emission depends on the Euler parameters.
  // Shift the scalar phase so that `decomposePsxGen` can emit an exact
  // (non-projective) decomposition in terms of `rz` and `sx`.
  const auto [theta, phi, lambda, phase] = paramsZyz(matrix);
  return {theta, phi, lambda, phase - (0.5 * (theta + phi + lambda))};
}

std::array<double, 4>
EulerDecomposition::paramsXzx(const Eigen::Matrix2cd& matrix) {
  // Rewrite the matrix into a form where the residual SU(2) part can be
  // interpreted as a Z-X-Z decomposition, then lift the resulting phase back
  // to the original matrix.
  auto det = matrix.determinant();
  auto phase = 0.5 * std::arg(det);
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

OneQubitGateSequence
EulerDecomposition::decomposeKAK(double theta, double phi, double lambda,
                                 double phase, GateKind kGate, GateKind aGate,
                                 bool simplify, std::optional<double> atol) {
  // Treat tiny angles as zero when simplification is enabled.
  double angleZeroEpsilon = atol.value_or(DEFAULT_ATOL);
  if (!simplify) {
    // setting atol to negative value to make all angle checks true; this will
    // effectively disable the simplification since all rotations appear to be
    // "necessary"
    angleZeroEpsilon = -1.0;
  }

  OneQubitGateSequence sequence{
      .gates = {},
      // Track the scalar phase so emitted K-A-K rotations match the input
      // unitary exactly (not only up to global phase).
      .globalPhase = phase - ((phi + lambda) / 2.),
  };
  if (std::abs(theta) <= angleZeroEpsilon) {
    // A(0) vanishes, so K(lambda) A(0) K(phi) collapses to K(lambda + phi).
    lambda += phi;
    lambda = helpers::mod2pi(lambda);
    if (std::abs(lambda) > angleZeroEpsilon) {
      sequence.gates.push_back({.type = kGate, .parameter = {lambda}});
      sequence.globalPhase += lambda / 2.0;
    }
    return sequence;
  }

  if (std::abs(theta - std::numbers::pi) <= angleZeroEpsilon) {
    // At theta ~= pi, Euler parameters are non-unique. Rewrite into a stable
    // equivalent form to keep emission deterministic.
    sequence.globalPhase += phi;
    lambda -= phi;
    phi = 0.0;
  }
  if (std::abs(helpers::mod2pi(lambda + std::numbers::pi)) <=
          angleZeroEpsilon ||
      std::abs(helpers::mod2pi(phi + std::numbers::pi)) <= angleZeroEpsilon) {
    // Shift away from the -pi branch cut by an equivalent parameterization.
    lambda += std::numbers::pi;
    theta = -theta;
    phi += std::numbers::pi;
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

OneQubitGateSequence
EulerDecomposition::decomposePsxGen(double theta, double phi, double lambda,
                                    double phase, bool allowXShortcut,
                                    bool simplify, std::optional<double> atol) {
  double angleZeroEpsilon = atol.value_or(DEFAULT_ATOL);
  if (!simplify) {
    // Disable all simplification checks by using a negative tolerance so that
    // every `std::abs(...) < atol` comparison evaluates to false.
    angleZeroEpsilon = -1.0;
  }

  OneQubitGateSequence sequence{
      .gates = {},
      .globalPhase = phase,
  };

  // Append `RZ(angle)` and add `angle / 2` to `globalPhase` so the combined
  // effect matches the `rz`/`sx` bookkeeping used here (RZ vs scalar phase).
  // Small angles after `mod2pi` are dropped when simplification is enabled.
  auto emitRzAsP = [&](double angle) {
    const double canonicalAngle = helpers::mod2pi(angle);
    if (std::abs(canonicalAngle) > angleZeroEpsilon) {
      sequence.gates.push_back(
          {.type = GateKind::RZ, .parameter = {canonicalAngle}});
      sequence.globalPhase += canonicalAngle / 2.0;
    }
  };

  // Zero-`sx` decomposition: RZ(phi) . I . RZ(lambda) collapses to a single
  // phase gate RZ(lambda + phi) (plus the matching phase correction).
  if (std::abs(theta) < angleZeroEpsilon) {
    emitRzAsP(lambda + phi);
    return sequence;
  }

  // Single-`sx` decomposition:
  //   RZ(phi) . RY(pi/2) . RZ(lambda)
  //     = P(phi + pi/2) . SX . P(lambda - pi/2) . e^{-i * pi / 4}
  if (std::abs(theta - (std::numbers::pi / 2.0)) < angleZeroEpsilon) {
    emitRzAsP(lambda - (std::numbers::pi / 2.0));
    sequence.gates.push_back({.type = GateKind::SX});
    emitRzAsP(phi + (std::numbers::pi / 2.0));
    return sequence;
  }

  // General two-`sx` decomposition.
  if (std::abs(theta - std::numbers::pi) < angleZeroEpsilon) {
    sequence.globalPhase += lambda;
    phi -= lambda;
    lambda = 0.0;
  }
  if (std::abs(helpers::mod2pi(lambda + std::numbers::pi)) < angleZeroEpsilon ||
      std::abs(helpers::mod2pi(phi)) < angleZeroEpsilon) {
    lambda += std::numbers::pi;
    theta = -theta;
    phi += std::numbers::pi;
    sequence.globalPhase -= theta;
  }
  // Shift theta and phi to turn the decomposition from
  //   RZ(phi) . RY(theta) . RZ(lambda)
  //     = RZ(phi) . RX(-pi/2) . RZ(theta) . RX(+pi/2) . RZ(lambda)
  // into P(phi + pi) . SX . P(theta + pi) . SX . P(lambda).
  theta += std::numbers::pi;
  phi += std::numbers::pi;
  sequence.globalPhase -= std::numbers::pi / 2.0;

  emitRzAsP(lambda);
  if (allowXShortcut && std::abs(helpers::mod2pi(theta)) < angleZeroEpsilon) {
    // `SX . P(theta) . SX` with `theta` congruent to `+/- pi` simplifies to
    // a bare `X` gate (up to the already-tracked global phase).
    sequence.gates.push_back({.type = GateKind::X});
  } else {
    sequence.gates.push_back({.type = GateKind::SX});
    emitRzAsP(theta);
    sequence.gates.push_back({.type = GateKind::SX});
  }
  emitRzAsP(phi);
  return sequence;
}

} // namespace mlir::qco::decomposition
