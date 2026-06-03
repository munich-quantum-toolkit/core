/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/IR/QCOUnitaryMatrixInterfaces.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <Eigen/Core>
#include <Eigen/LU>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <optional>

namespace mlir::qco::decomposition {

//===----------------------------------------------------------------------===//
// Euler decomposition (angles)
//===----------------------------------------------------------------------===//

std::array<double, 4>
EulerDecomposition::anglesFromUnitary(const Eigen::Matrix2cd& matrix,
                                      EulerBasis basis) {
  switch (basis) {
  case EulerBasis::XYX:
    return paramsXYX(matrix);
  case EulerBasis::XZX:
    return paramsXZX(matrix);
  case EulerBasis::ZYZ:
    return paramsZYZ(matrix);
  case EulerBasis::ZXZ:
    return paramsZXZ(matrix);
  case EulerBasis::U:
    // The `u` gate parameterization is derived from the standard Z-Y-Z form.
    return paramsZYZ(matrix);
  case EulerBasis::ZSXX:
    return paramsPSX(matrix);
  }
  llvm::reportFatalInternalError(
      "Unsupported Euler basis for angle computation in decomposition!");
}

std::array<double, 4>
EulerDecomposition::paramsZYZ(const Eigen::Matrix2cd& matrix) {
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
  const auto lambda = ang1 - ang2;
  return {theta, phi, lambda, phase};
}

std::array<double, 4>
EulerDecomposition::paramsZXZ(const Eigen::Matrix2cd& matrix) {
  // Convert from the Z-Y-Z parameterization via the standard basis-change
  // identity `ry(a) = rz(pi/2) · rx(a) · rz(-pi/2)`, i.e.
  // `rz(phi) · ry(theta) · rz(lambda) =
  // rz(phi + pi/2) · rx(theta) · rz(lambda - pi/2)`.
  const auto [theta, phi, lambda, phase] = paramsZYZ(matrix);
  return {theta, phi + (std::numbers::pi / 2.0),
          lambda - (std::numbers::pi / 2.0), phase};
}

std::array<double, 4>
EulerDecomposition::paramsXYX(const Eigen::Matrix2cd& matrix) {
  // Conjugating by Hadamards transforms an X-Y-X decomposition problem into a
  // Z-Y-Z one, so we solve it there and map the angles back.
  const Eigen::Matrix2cd matZYZ{
      {0.5 * (matrix(0, 0) + matrix(0, 1) + matrix(1, 0) + matrix(1, 1)),
       0.5 * (matrix(0, 0) - matrix(0, 1) + matrix(1, 0) - matrix(1, 1))},
      {0.5 * (matrix(0, 0) + matrix(0, 1) - matrix(1, 0) - matrix(1, 1)),
       0.5 * (matrix(0, 0) - matrix(0, 1) - matrix(1, 0) + matrix(1, 1))},
  };
  auto [theta, phi, lambda, phase] = paramsZYZ(matZYZ);
  auto newPhi = helpers::mod2pi(phi + std::numbers::pi, 0.);
  auto newLambda = helpers::mod2pi(lambda + std::numbers::pi, 0.);
  return {
      theta,
      newPhi,
      newLambda,
      phase + ((newPhi + newLambda - phi - lambda) / 2.),
  };
}

std::array<double, 4>
EulerDecomposition::paramsPSX(const Eigen::Matrix2cd& matrix) {
  const auto [theta, phi, lambda, phase] = paramsZYZ(matrix);
  return {theta, phi, lambda, phase - (0.5 * (theta + phi + lambda))};
}

std::array<double, 4>
EulerDecomposition::paramsXZX(const Eigen::Matrix2cd& matrix) {
  // Rewrite the matrix into a form where the residual SU(2) part can be
  // interpreted as a Z-X-Z decomposition, then lift the resulting phase back
  // to the original matrix.
  auto det = matrix.determinant();
  auto phase = 0.5 * std::arg(det);
  auto sqrtDet = std::sqrt(det);
  const Eigen::Matrix2cd matZXZ{
      {
          {(matrix(0, 0) / sqrtDet).real(), (matrix(1, 0) / sqrtDet).imag()},
          {(matrix(1, 0) / sqrtDet).real(), (matrix(0, 0) / sqrtDet).imag()},
      },
      {
          {-(matrix(1, 0) / sqrtDet).real(), (matrix(0, 0) / sqrtDet).imag()},
          {(matrix(0, 0) / sqrtDet).real(), -(matrix(1, 0) / sqrtDet).imag()},
      },
  };
  auto [theta, phi, lambda, phaseZXZ] = paramsZXZ(matZXZ);
  return {theta, phi, lambda, phase + phaseZXZ};
}

//===----------------------------------------------------------------------===//
// Euler synthesis (IR emission)
//===----------------------------------------------------------------------===//

static void emitGPhaseIfNeeded(OpBuilder& builder, Location loc, double phase) {
  if (std::abs(phase) <= mlir::utils::TOLERANCE) {
    return;
  }
  GPhaseOp::create(builder, loc, phase);
}

static double phaseToMatchTarget(const Eigen::Matrix2cd& target,
                                 const Eigen::Matrix2cd& emitted) {
  // If `target == s * emitted`, then `target * emitted^H == s * I`.
  const auto s = (target * emitted.adjoint())(0, 0);
  return std::arg(s);
}

static void accumulateEmittedMatrix(UnitaryMatrixOpInterface op,
                                    Eigen::Matrix2cd& emitted) {
  Eigen::Matrix2cd m;
  if (!op.getUnitaryMatrix2x2(m)) {
    llvm::reportFatalInternalError(
        "Expected emitted 1q op to have constant unitary matrix");
  }

  // Execution order: first emitted op applied first => multiply on the left.
  emitted = m * emitted;
}

static Value emitRZ(OpBuilder& builder, Location loc, Value qubit, double angle,
                    Eigen::Matrix2cd& emitted) {
  auto op = RZOp::create(builder, loc, qubit, angle);
  accumulateEmittedMatrix(op, emitted);
  return op.getQubitOut();
}

static Value emitRY(OpBuilder& builder, Location loc, Value qubit, double angle,
                    Eigen::Matrix2cd& emitted) {
  auto op = RYOp::create(builder, loc, qubit, angle);
  accumulateEmittedMatrix(op, emitted);
  return op.getQubitOut();
}

static Value emitRX(OpBuilder& builder, Location loc, Value qubit, double angle,
                    Eigen::Matrix2cd& emitted) {
  auto op = RXOp::create(builder, loc, qubit, angle);
  accumulateEmittedMatrix(op, emitted);
  return op.getQubitOut();
}

static Value emitSX(OpBuilder& builder, Location loc, Value qubit,
                    Eigen::Matrix2cd& emitted) {
  auto op = SXOp::create(builder, loc, qubit);
  accumulateEmittedMatrix(op, emitted);
  return op.getQubitOut();
}

static Value emitX(OpBuilder& builder, Location loc, Value qubit,
                   Eigen::Matrix2cd& emitted) {
  auto op = XOp::create(builder, loc, qubit);
  accumulateEmittedMatrix(op, emitted);
  return op.getQubitOut();
}

static Value emitU(OpBuilder& builder, Location loc, Value qubit, double theta,
                   double phi, double lambda, Eigen::Matrix2cd& emitted) {
  auto op = UOp::create(builder, loc, qubit, theta, phi, lambda);
  accumulateEmittedMatrix(op, emitted);
  return op.getQubitOut();
}

static Value emitKAK(OpBuilder& builder, Location loc, Value qubit,
                     const Eigen::Matrix2cd& targetMatrix, double theta,
                     double phi, double lambda, EulerBasis basis) {
  Eigen::Matrix2cd emitted = Eigen::Matrix2cd::Identity();

  auto emitK = [&](double a) {
    const double canonical = helpers::mod2pi(a, mlir::utils::TOLERANCE);
    switch (basis) {
    case EulerBasis::ZYZ:
    case EulerBasis::ZXZ:
      qubit = emitRZ(builder, loc, qubit, canonical, emitted);
      break;
    case EulerBasis::XZX:
    case EulerBasis::XYX:
      qubit = emitRX(builder, loc, qubit, canonical, emitted);
      break;
    default:
      llvm::reportFatalInternalError("Invalid K gate for KAK emission");
    }
  };

  auto emitA = [&](double a) {
    switch (basis) {
    case EulerBasis::ZYZ:
    case EulerBasis::XYX:
      qubit = emitRY(builder, loc, qubit, a, emitted);
      break;
    case EulerBasis::ZXZ:
      qubit = emitRX(builder, loc, qubit, a, emitted);
      break;
    case EulerBasis::XZX:
      qubit = emitRZ(builder, loc, qubit, a, emitted);
      break;
    default:
      llvm::reportFatalInternalError("Invalid A gate for KAK emission");
    }
  };

  emitK(lambda);
  emitA(theta);
  emitK(phi);
  emitGPhaseIfNeeded(builder, loc, phaseToMatchTarget(targetMatrix, emitted));
  return qubit;
}

static Value emitPSXGen(OpBuilder& builder, Location loc, Value qubit,
                        const Eigen::Matrix2cd& targetMatrix, double theta,
                        double phi, double lambda) {
  constexpr double eps = mlir::utils::TOLERANCE;
  Eigen::Matrix2cd emitted = Eigen::Matrix2cd::Identity();

  auto emitCanonicalRZ = [&](double angle) {
    const double canonicalAngle = helpers::mod2pi(angle, eps);
    qubit = emitRZ(builder, loc, qubit, canonicalAngle, emitted);
  };

  if (std::abs(theta - (std::numbers::pi / 2.0)) < eps) {
    emitCanonicalRZ(lambda - (std::numbers::pi / 2.0));
    qubit = emitSX(builder, loc, qubit, emitted);
    emitCanonicalRZ(phi + (std::numbers::pi / 2.0));
    emitGPhaseIfNeeded(builder, loc, phaseToMatchTarget(targetMatrix, emitted));
    return qubit;
  }

  // General double-`sx` case: reparameterize angles, then fix global phase from
  // the accumulated unitary.
  if (std::abs(theta - std::numbers::pi) < eps) {
    phi -= lambda;
    lambda = 0.0;
  }
  if (std::abs(helpers::mod2pi(lambda + std::numbers::pi, eps)) < eps ||
      std::abs(helpers::mod2pi(phi, eps)) < eps) {
    lambda += std::numbers::pi;
    theta = -theta;
    phi += std::numbers::pi;
  }

  theta += std::numbers::pi;
  phi += std::numbers::pi;

  emitCanonicalRZ(lambda);

  // `sx · rz(θ) · sx` equals `x` when θ ≡ 0 (mod 2π).
  if (std::abs(helpers::mod2pi(theta, eps)) < eps) {
    qubit = emitX(builder, loc, qubit, emitted);
  } else {
    qubit = emitSX(builder, loc, qubit, emitted);
    emitCanonicalRZ(theta);
    qubit = emitSX(builder, loc, qubit, emitted);
  }

  emitCanonicalRZ(phi);
  emitGPhaseIfNeeded(builder, loc, phaseToMatchTarget(targetMatrix, emitted));
  return qubit;
}

std::optional<EulerBasis> parseEulerBasis(StringRef basis) {
  const auto b = basis.lower();
  if (b == "zyz") {
    return EulerBasis::ZYZ;
  }
  if (b == "zxz") {
    return EulerBasis::ZXZ;
  }
  if (b == "xzx") {
    return EulerBasis::XZX;
  }
  if (b == "xyx") {
    return EulerBasis::XYX;
  }
  if (b == "u") {
    return EulerBasis::U;
  }
  if (b == "zsxx") {
    return EulerBasis::ZSXX;
  }
  return std::nullopt;
}

Value synthesizeUnitary1QEuler(OpBuilder& builder, Location loc, Value qubit,
                               const Eigen::Matrix2cd& targetMatrix,
                               EulerBasis basis) {
  const auto [theta, phi, lambda, phase] =
      EulerDecomposition::anglesFromUnitary(targetMatrix, basis);

  switch (basis) {
  case EulerBasis::ZYZ:
  case EulerBasis::ZXZ:
  case EulerBasis::XZX:
  case EulerBasis::XYX:
    qubit =
        emitKAK(builder, loc, qubit, targetMatrix, theta, phi, lambda, basis);
    break;
  case EulerBasis::U: {
    Eigen::Matrix2cd emitted = Eigen::Matrix2cd::Identity();
    qubit = emitU(builder, loc, qubit, theta, phi, lambda, emitted);
    emitGPhaseIfNeeded(builder, loc, phaseToMatchTarget(targetMatrix, emitted));
    break;
  }
  case EulerBasis::ZSXX:
    qubit = emitPSXGen(builder, loc, qubit, targetMatrix, theta, phi, lambda);
    break;
  }

  (void)phase;
  return qubit;
}

} // namespace mlir::qco::decomposition
