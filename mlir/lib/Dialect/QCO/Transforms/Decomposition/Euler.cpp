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
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>

namespace mlir::qco::decomposition {

/**
 * @brief Wraps `angle` into `[-pi, pi)`, mapping `+pi` (within tolerance) to
 * `-pi`.
 *
 * @param angle The angle to wrap, in radians.
 * @return The wrapped angle in `[-pi, pi)`.
 */
[[nodiscard]] static double mod2pi(const double angle) {
  if (!std::isfinite(angle)) {
    return angle;
  }

  constexpr double pi = std::numbers::pi;
  constexpr double twoPi = 2.0 * std::numbers::pi;

  double r = std::fmod(angle + pi, twoPi);
  if (r < 0.0) {
    r += twoPi;
  }
  double wrapped = r - pi;

  if (wrapped >= pi - utils::TOLERANCE) {
    wrapped = -pi;
  }

  return wrapped;
}

/**
 * @brief Conjugates a single-qubit matrix by Hadamard (`H * m * H`).
 *
 * Maps XYX / XZX parameterizations to ZYZ / ZXZ.
 *
 * @param m The single-qubit matrix to conjugate.
 * @return `H * m * H`.
 */
[[nodiscard]] static Matrix2x2 hadamardConjugate(const Matrix2x2& m) {
  const auto a = m(0, 0);
  const auto b = m(0, 1);
  const auto c = m(1, 0);
  const auto d = m(1, 1);
  return Matrix2x2::fromElements(0.5 * (a + b + c + d), 0.5 * (a - b + c - d),
                                 0.5 * (a + b - c - d), 0.5 * (a - b - c + d));
}

/**
 * @brief Whether `angle` is numerically zero for gate-emission purposes.
 *
 * @param angle Rotation angle in radians.
 * @return `true` when no rotation gate should be emitted.
 */
[[nodiscard]] static bool isNearZeroRotationAngle(const double angle) {
  return std::abs(angle) <= utils::TOLERANCE;
}

void emitGPhaseIfNeeded(OpBuilder& builder, Location loc, const double phase) {
  if (isNearZeroRotationAngle(mod2pi(phase))) {
    return;
  }
  GPhaseOp::create(builder, loc, phase);
}

//===----------------------------------------------------------------------===//
// Euler decomposition (angles)
//===----------------------------------------------------------------------===//

/**
 * @brief Z-Y-Z Euler angles and global phase for a 2x2 unitary.
 *
 * @param matrix Single-qubit unitary to decompose.
 * @return Z-Y-Z angles and global phase.
 */
[[nodiscard]] static EulerAngles paramsZYZ(const Matrix2x2& matrix) {
  // det(U) = exp(2i*phase)
  const Complex det = matrix.determinant();
  const auto detArg = std::arg(det);
  const auto phase = 0.5 * detArg;
  const auto theta =
      2. * std::atan2(std::abs(matrix(1, 0)), std::abs(matrix(0, 0)));
  const auto ang1 = std::arg(matrix(1, 1));
  double ang2 = 0.0;
  if (std::abs(matrix(1, 0)) > utils::TOLERANCE) {
    ang2 = std::arg(matrix(1, 0));
  } else if (std::abs(matrix(0, 1)) > utils::TOLERANCE) {
    ang2 = std::arg(matrix(0, 1));
  }
  const auto phi = ang1 + ang2 - detArg;
  const auto lambda = ang1 - ang2;
  return {.theta = theta, .phi = phi, .lambda = lambda, .phase = phase};
}

/**
 * @brief Z-X-Z Euler angles via `RY(theta) = RZ(pi/2)*RX(theta)*RZ(-pi/2)`.
 *
 * @param matrix Single-qubit unitary to decompose.
 * @return Z-X-Z angles and global phase.
 */
[[nodiscard]] static EulerAngles paramsZXZ(const Matrix2x2& matrix) {
  const auto [theta, phi, lambda, phase] = paramsZYZ(matrix);
  return {.theta = theta,
          .phi = phi + (std::numbers::pi / 2.0),
          .lambda = lambda - (std::numbers::pi / 2.0),
          .phase = phase};
}

/**
 * @brief X-Z-X Euler angles (Z-X-Z under H conjugation).
 *
 * @param matrix Single-qubit unitary to decompose.
 * @return X-Z-X angles and global phase.
 */
[[nodiscard]] static EulerAngles paramsXZX(const Matrix2x2& matrix) {
  return paramsZXZ(hadamardConjugate(matrix));
}

/**
 * @brief X-Y-X Euler angles via `H*RY(theta)*H = RY(-theta)`.
 *
 * @param matrix Single-qubit unitary to decompose.
 * @return X-Y-X angles and global phase.
 */
[[nodiscard]] static EulerAngles paramsXYX(const Matrix2x2& matrix) {
  // Shift outer angles by pi and fix global phase.
  const auto [theta, phi, lambda, phase] = paramsZYZ(hadamardConjugate(matrix));
  return {.theta = theta,
          .phi = phi + std::numbers::pi,
          .lambda = lambda + std::numbers::pi,
          .phase = phase + std::numbers::pi};
}

/**
 * @brief `U`-basis angles (Z-Y-Z angles with a `U`-vs-`RZ*RY*RZ` phase fix).
 *
 * @param matrix Single-qubit unitary to decompose.
 * @return `U`-gate angles and global phase.
 */
[[nodiscard]] static EulerAngles paramsU(const Matrix2x2& matrix) {
  // `U` differs from RZ(phi)*RY(theta)*RZ(lambda) by a global phase of
  // -(phi + lambda)/2.
  const auto [theta, phi, lambda, phase] = paramsZYZ(matrix);
  return {.theta = theta,
          .phi = phi,
          .lambda = lambda,
          .phase = phase - (0.5 * (phi + lambda))};
}

EulerAngles anglesFromUnitary(const Matrix2x2& matrix, const EulerBasis basis) {
  switch (basis) {
  case EulerBasis::ZYZ:
  case EulerBasis::ZSXX:
    return paramsZYZ(matrix);
  case EulerBasis::ZXZ:
    return paramsZXZ(matrix);
  case EulerBasis::XZX:
    return paramsXZX(matrix);
  case EulerBasis::XYX:
  case EulerBasis::R:
    // The `R` basis reuses the X-Y-X angles and lowers `Rx`/`Ry` to the native
    // `R(theta, phi)` gate (`Rx(a) == R(a, 0)`, `Ry(a) == R(a, pi/2)`).
    return paramsXYX(matrix);
  case EulerBasis::U:
    return paramsU(matrix);
  default:
    llvm_unreachable("invalid Euler basis");
  }
}

//===----------------------------------------------------------------------===//
// Euler synthesis (plan + emit)
//===----------------------------------------------------------------------===//

namespace {

/**
 * @brief One gate in a planned single-qubit synthesis sequence.
 *
 * `RZ`/`RY`/`RX` use @p theta as the rotation angle; `U` uses all three angles.
 */
struct SynthesisStep {
  enum class Kind : std::uint8_t { RZ, RY, RX, SX, X, U, R };

  Kind kind = Kind::RZ;
  double theta = 0.0;
  double phi = 0.0;
  double lambda = 0.0;
};

/** @brief Planned single-qubit Euler synthesis (gate list + optional `gphase`).
 */
struct Unitary1QEulerPlan {
  SmallVector<SynthesisStep, 5> steps;
  double phase = 0.0;

  /// @brief Number of native gates in the planned sequence (excludes `gphase`).
  [[nodiscard]] std::size_t gateCount() const { return steps.size(); }

  /**
   * @brief Appends a rotation step for non-negligible angles.
   *
   * @param kind The rotation axis (RZ/RY/RX)
   * @param angle The rotation angle in radians.
   */
  void appendRotation(const SynthesisStep::Kind kind, const double angle) {
    if (!isNearZeroRotationAngle(angle)) {
      steps.emplace_back(kind, angle);
    }
  }

  /**
   * @brief Appends a native `R(angle, axis)` step for non-negligible angles.
   *
   * @param angle The rotation angle in radians.
   * @param axis The rotation axis in the XY-plane (`0` for `Rx`, `pi/2` for
   *             `Ry`).
   */
  void appendRStep(const double angle, const double axis) {
    if (!isNearZeroRotationAngle(angle)) {
      steps.emplace_back(SynthesisStep::Kind::R, angle, axis);
    }
  }

  /**
   * @brief Appends the decomposition for @p basis based on @p angles.
   *
   * @param angles The angles to use for the decomposition.
   * @param basis The basis to use for the decomposition.
   */
  void appendDecomposition(const EulerAngles& angles, const EulerBasis basis) {
    if (isNearZeroRotationAngle(angles.theta) &&
        isNearZeroRotationAngle(angles.phi) &&
        isNearZeroRotationAngle(angles.lambda)) {
      phase = angles.phase;
      return;
    }

    if (isNearZeroRotationAngle(angles.theta)) {
      switch (basis) {
      case EulerBasis::ZYZ:
      case EulerBasis::ZXZ:
      case EulerBasis::ZSXX:
        appendRotation(SynthesisStep::Kind::RZ, angles.phi + angles.lambda);
        break;

      case EulerBasis::XZX:
      case EulerBasis::XYX:
        appendRotation(SynthesisStep::Kind::RX, angles.phi + angles.lambda);
        break;
      case EulerBasis::R:
        appendRStep(angles.phi + angles.lambda, 0.0);
        break;
      case EulerBasis::U:
        steps.emplace_back(SynthesisStep::Kind::U, 0.0, angles.phi,
                           angles.lambda);
        break;
      }
      phase = angles.phase;
      return;
    }

    switch (basis) {
    case EulerBasis::ZYZ:
      appendRotation(SynthesisStep::Kind::RZ, angles.lambda);
      steps.emplace_back(SynthesisStep::Kind::RY, angles.theta);
      appendRotation(SynthesisStep::Kind::RZ, angles.phi);
      phase = angles.phase;
      break;
    case EulerBasis::ZXZ:
      appendRotation(SynthesisStep::Kind::RZ, angles.lambda);
      steps.emplace_back(SynthesisStep::Kind::RX, angles.theta);
      appendRotation(SynthesisStep::Kind::RZ, angles.phi);
      phase = angles.phase;
      break;
    case EulerBasis::XZX:
      appendRotation(SynthesisStep::Kind::RX, angles.lambda);
      steps.emplace_back(SynthesisStep::Kind::RZ, angles.theta);
      appendRotation(SynthesisStep::Kind::RX, angles.phi);
      phase = angles.phase;
      break;
    case EulerBasis::XYX:
      appendRotation(SynthesisStep::Kind::RX, angles.lambda);
      steps.emplace_back(SynthesisStep::Kind::RY, angles.theta);
      appendRotation(SynthesisStep::Kind::RX, angles.phi);
      phase = angles.phase;
      break;
    case EulerBasis::R:
      // X-Y-X with `Rx(a) == R(a, 0)` and `Ry(a) == R(a, pi/2)`.
      appendRStep(angles.lambda, 0.0);
      steps.emplace_back(SynthesisStep::Kind::R, angles.theta,
                         std::numbers::pi / 2.0);
      appendRStep(angles.phi, 0.0);
      phase = angles.phase;
      break;
    case EulerBasis::U:
      steps.emplace_back(SynthesisStep::Kind::U, angles.theta, angles.phi,
                         angles.lambda);
      phase = angles.phase;
      break;
    case EulerBasis::ZSXX: {
      constexpr double pi = std::numbers::pi;
      constexpr double halfPi = std::numbers::pi / 2.0;
      constexpr double quarterPi = std::numbers::pi / 4.0;

      if (isNearZeroRotationAngle(angles.theta - halfPi)) {
        appendRotation(SynthesisStep::Kind::RZ, angles.lambda - halfPi);
        steps.emplace_back(SynthesisStep::Kind::SX);
        appendRotation(SynthesisStep::Kind::RZ, angles.phi + halfPi);
        phase = angles.phase - quarterPi;
        return;
      }

      appendRotation(SynthesisStep::Kind::RZ, angles.lambda);
      if (isNearZeroRotationAngle(angles.theta - pi)) {
        steps.emplace_back(SynthesisStep::Kind::X);
        phase = angles.phase - halfPi;
      } else {
        steps.emplace_back(SynthesisStep::Kind::SX);
        appendRotation(SynthesisStep::Kind::RZ, angles.theta + pi);
        steps.emplace_back(SynthesisStep::Kind::SX);
        phase = angles.phase + halfPi;
      }
      appendRotation(SynthesisStep::Kind::RZ, angles.phi + pi);
      break;
    }
    }
  }
};
} // namespace

/**
 * @brief Builds a gate plan for @p targetMatrix in @p basis without emitting
 * IR.
 *
 * @param targetMatrix Single-qubit unitary to synthesize.
 * @param basis Native gate basis.
 * @return Planned gate sequence and optional global phase.
 */
[[nodiscard]] static Unitary1QEulerPlan
planUnitary1QEuler(const Matrix2x2& targetMatrix, const EulerBasis basis) {
  Unitary1QEulerPlan plan;
  if (targetMatrix.isApprox(Matrix2x2::identity())) {
    return plan;
  }

  const EulerAngles angles = anglesFromUnitary(targetMatrix, basis);
  plan.appendDecomposition(angles, basis);
  return plan;
}

/**
 * @brief Emits the gates described by @p plan and returns the output qubit.
 *
 * @param builder Builder for the emitted operations.
 * @param loc Location for the emitted operations.
 * @param qubit Input qubit value.
 * @param plan Precomputed synthesis plan.
 * @return Qubit value after all planned gates (and `gphase` when needed).
 */
[[nodiscard]] static Value
emitUnitary1QEulerPlan(OpBuilder& builder, Location loc, Value qubit,
                       const Unitary1QEulerPlan& plan) {
  for (const auto& [kind, theta, phi, lambda] : plan.steps) {
    switch (kind) {
    case SynthesisStep::Kind::RZ:
      qubit = RZOp::create(builder, loc, qubit, theta).getQubitOut();
      break;
    case SynthesisStep::Kind::RY:
      qubit = RYOp::create(builder, loc, qubit, theta).getQubitOut();
      break;
    case SynthesisStep::Kind::RX:
      qubit = RXOp::create(builder, loc, qubit, theta).getQubitOut();
      break;
    case SynthesisStep::Kind::SX:
      qubit = SXOp::create(builder, loc, qubit).getQubitOut();
      break;
    case SynthesisStep::Kind::X:
      qubit = XOp::create(builder, loc, qubit).getQubitOut();
      break;
    case SynthesisStep::Kind::U:
      qubit =
          UOp::create(builder, loc, qubit, theta, phi, lambda).getQubitOut();
      break;
    case SynthesisStep::Kind::R:
      qubit = ROp::create(builder, loc, qubit, theta, phi).getQubitOut();
      break;
    }
  }
  emitGPhaseIfNeeded(builder, loc, plan.phase);
  return qubit;
}

std::optional<EulerBasis> parseEulerBasis(StringRef basis) {
  return StringSwitch<std::optional<EulerBasis>>(basis.lower())
      .Case("zyz", EulerBasis::ZYZ)
      .Case("zxz", EulerBasis::ZXZ)
      .Case("xzx", EulerBasis::XZX)
      .Case("xyx", EulerBasis::XYX)
      .Case("u", EulerBasis::U)
      .Case("zsxx", EulerBasis::ZSXX)
      .Case("r", EulerBasis::R)
      .Default(std::nullopt);
}

std::optional<Value>
synthesizeUnitary1QEuler(OpBuilder& builder, Location loc, Value qubit,
                         const Matrix2x2& composed, const std::size_t runSize,
                         const bool hasNonBasisGate, const EulerBasis basis) {
  const Unitary1QEulerPlan plan = planUnitary1QEuler(composed, basis);
  if (!hasNonBasisGate && runSize <= plan.gateCount()) {
    return std::nullopt;
  }
  return emitUnitary1QEulerPlan(builder, loc, qubit, plan);
}

} // namespace mlir::qco::decomposition
