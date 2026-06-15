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

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <array>
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
[[nodiscard]] static bool isNearZeroRotationAngle(double angle) {
  return std::abs(angle) <= utils::TOLERANCE;
}

/**
 * @brief Emits `qco.gphase` when `phase` is outside tolerance.
 *
 * @param builder Builder for the operation.
 * @param loc Location of the operation.
 * @param phase Global phase in radians.
 */
static void emitGPhaseIfNeeded(OpBuilder& builder, Location loc, double phase) {
  if (isNearZeroRotationAngle(phase)) {
    return;
  }
  GPhaseOp::create(builder, loc, phase);
}

//===----------------------------------------------------------------------===//
// Euler decomposition (angles)
//===----------------------------------------------------------------------===//

/**
 * @brief Euler angles `(theta, phi, lambda)` and global phase for a 2x2
 * unitary.
 */
namespace {

struct EulerAngles {
  double theta = 0.0;  ///< Middle rotation angle.
  double phi = 0.0;    ///< First outer rotation angle.
  double lambda = 0.0; ///< Second outer rotation angle.
  double phase = 0.0;  ///< Global phase in radians.
};

} // namespace

/**
 * @brief Z-Y-Z Euler angles and global phase for a 2x2 unitary.
 *
 * @param matrix Single-qubit unitary to decompose.
 * @return Z-Y-Z angles and global phase.
 */
[[nodiscard]] static EulerAngles paramsZYZ(const Matrix2x2& matrix) {
  // det(U) = exp(2i*phase); invert the Z-Y-Z parameterization of U's entries.
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
 * @brief X-Z-X Euler angles (Z-X-Z under H conjugation, no Y sign flip).
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
  const auto newPhi = mod2pi(phi + std::numbers::pi);
  const auto newLambda = mod2pi(lambda + std::numbers::pi);
  return {.theta = theta,
          .phi = newPhi,
          .lambda = newLambda,
          .phase = phase + ((newPhi + newLambda - phi - lambda) / 2.)};
}

/**
 * @brief `U`-basis angles (Z-Y-Z angles with a `U`-vs-`RZ·RY·RZ` phase fix).
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

/**
 * @brief Extracts `(theta, phi, lambda, phase)` for KAK and `U` bases.
 *
 * @param matrix The single-qubit unitary to decompose.
 * @param basis The target Euler basis.
 * @return The extracted Euler angles and global phase.
 */
[[nodiscard]] static EulerAngles anglesFromUnitary(const Matrix2x2& matrix,
                                                   const EulerBasis basis) {
  switch (basis) {
  case EulerBasis::ZYZ:
    return paramsZYZ(matrix);
  case EulerBasis::ZXZ:
    return paramsZXZ(matrix);
  case EulerBasis::XZX:
    return paramsXZX(matrix);
  case EulerBasis::XYX:
    return paramsXYX(matrix);
  case EulerBasis::U:
    return paramsU(matrix);
  default:
    llvm::reportFatalInternalError(
        "Unsupported Euler basis for angle computation in decomposition!");
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
  enum class Kind : std::uint8_t { RZ, RY, RX, SX, X, U };

  Kind kind = Kind::RZ;
  double theta = 0.0;
  double phi = 0.0;
  double lambda = 0.0;
};

/** @brief Planned single-qubit Euler synthesis (gate list + optional `gphase`).
 */
struct Unitary1QEulerPlan {
  llvm::SmallVector<SynthesisStep, 5> steps;
  double phase = 0.0;

  /// @brief Number of native gates in the planned sequence (excludes `gphase`).
  [[nodiscard]] std::size_t gateCount() const { return steps.size(); }
};

} // namespace

/**
 * @brief Appends a rotation step when @p angle is outside tolerance.
 *
 * @param steps Planned gate sequence to extend.
 * @param kind Rotation axis (`RZ`, `RY`, or `RX`).
 * @param angle Rotation angle in radians.
 */
static void appendRotationIf(llvm::SmallVectorImpl<SynthesisStep>& steps,
                             const SynthesisStep::Kind kind,
                             const double angle) {
  if (!isNearZeroRotationAngle(angle)) {
    steps.push_back({.kind = kind, .theta = angle});
  }
}

/**
 * @brief Appends the three KAK rotations for @p basis to @p steps.
 *
 * Uses @p angles as outer–middle–outer rotations
 * (`K(phi) * A(theta) * K(lambda)` with axes from @p basis).
 *
 * @param steps Planned gate sequence to extend.
 * @param angles Decomposed Euler angles and global phase.
 * @param basis Target KAK basis (`ZYZ`, `ZXZ`, `XZX`, or `XYX`).
 */
static void appendKAKSteps(llvm::SmallVectorImpl<SynthesisStep>& steps,
                           const EulerAngles& angles, const EulerBasis basis) {
  using Kind = SynthesisStep::Kind;
  // Outer (K) and middle (A) rotation axes per KAK basis.
  struct KAKAxes {
    Kind outer;
    Kind middle;
  };
  const auto axes = [&]() -> KAKAxes {
    switch (basis) {
    case EulerBasis::ZYZ:
      return {.outer = Kind::RZ, .middle = Kind::RY};
    case EulerBasis::ZXZ:
      return {.outer = Kind::RZ, .middle = Kind::RX};
    case EulerBasis::XZX:
      return {.outer = Kind::RX, .middle = Kind::RZ};
    case EulerBasis::XYX:
      return {.outer = Kind::RX, .middle = Kind::RY};
    default:
      llvm::reportFatalInternalError("Invalid Euler basis for KAK planning");
    }
  }();

  appendRotationIf(steps, axes.outer, angles.lambda);
  appendRotationIf(steps, axes.middle, angles.theta);
  appendRotationIf(steps, axes.outer, angles.phi);
}

/**
 * @brief Fills @p plan with an `RZ` / `SX` / `X` gate sequence from Z-Y-Z
 * angles.
 *
 * Implements the canonical ZSXX synthesis cases (identity, `theta = 0`,
 * `theta = pi/2`, `theta = pi`, and general) and sets @p plan.phase for any
 * global-phase correction.
 *
 * @param plan Synthesis plan to populate.
 * @param zyz Z-Y-Z Euler angles of the target unitary.
 */
static void planZSXX(Unitary1QEulerPlan& plan, const EulerAngles& zyz) {
  constexpr double pi = std::numbers::pi;
  constexpr double halfPi = std::numbers::pi / 2.0;
  constexpr double quarterPi = std::numbers::pi / 4.0;

  const auto theta = zyz.theta;
  const auto phi = zyz.phi;
  const auto lambda = zyz.lambda;
  const auto pushRZ = [&](const double angle) {
    appendRotationIf(plan.steps, SynthesisStep::Kind::RZ, angle);
  };
  const auto pushSX = [&] {
    plan.steps.push_back({.kind = SynthesisStep::Kind::SX});
  };
  const auto pushX = [&] {
    plan.steps.push_back({.kind = SynthesisStep::Kind::X});
  };

  if (isNearZeroRotationAngle(theta) && isNearZeroRotationAngle(phi) &&
      isNearZeroRotationAngle(lambda)) {
    plan.phase = zyz.phase;
    return;
  }

  if (isNearZeroRotationAngle(theta)) {
    pushRZ(lambda);
    pushRZ(phi);
    plan.phase = zyz.phase;
    return;
  }

  if (isNearZeroRotationAngle(theta - halfPi)) {
    pushRZ(lambda - halfPi);
    pushSX();
    pushRZ(phi + halfPi);
    plan.phase = zyz.phase - quarterPi;
    return;
  }

  if (isNearZeroRotationAngle(theta - pi)) {
    pushRZ(lambda);
    pushX();
    pushRZ(phi + pi);
    plan.phase = zyz.phase - halfPi;
    return;
  }

  pushRZ(lambda);
  pushSX();
  pushRZ(theta + pi);
  pushSX();
  pushRZ(phi + pi);
  plan.phase = zyz.phase + halfPi;
}

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

  if (basis == EulerBasis::ZSXX) {
    planZSXX(plan, anglesFromUnitary(targetMatrix, EulerBasis::ZYZ));
    return plan;
  }

  const EulerAngles angles = anglesFromUnitary(targetMatrix, basis);
  plan.phase = angles.phase;

  if (basis == EulerBasis::U) {
    plan.steps.push_back({.kind = SynthesisStep::Kind::U,
                          .theta = angles.theta,
                          .phi = angles.phi,
                          .lambda = angles.lambda});
    return plan;
  }

  appendKAKSteps(plan.steps, angles, basis);
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
  for (const SynthesisStep& step : plan.steps) {
    switch (step.kind) {
    case SynthesisStep::Kind::RZ:
      qubit = RZOp::create(builder, loc, qubit, step.theta).getQubitOut();
      break;
    case SynthesisStep::Kind::RY:
      qubit = RYOp::create(builder, loc, qubit, step.theta).getQubitOut();
      break;
    case SynthesisStep::Kind::RX:
      qubit = RXOp::create(builder, loc, qubit, step.theta).getQubitOut();
      break;
    case SynthesisStep::Kind::SX:
      qubit = SXOp::create(builder, loc, qubit).getQubitOut();
      break;
    case SynthesisStep::Kind::X:
      qubit = XOp::create(builder, loc, qubit).getQubitOut();
      break;
    case SynthesisStep::Kind::U:
      qubit =
          UOp::create(builder, loc, qubit, step.theta, step.phi, step.lambda)
              .getQubitOut();
      break;
    }
  }
  emitGPhaseIfNeeded(builder, loc, plan.phase);
  return qubit;
}

std::optional<EulerBasis> parseEulerBasis(StringRef basis) {
  struct EulerBasisName {
    const char* name;
    EulerBasis value;
  };
  constexpr std::array<EulerBasisName, 6> eulerBasisTable{{
      {.name = "zyz", .value = EulerBasis::ZYZ},
      {.name = "zxz", .value = EulerBasis::ZXZ},
      {.name = "xzx", .value = EulerBasis::XZX},
      {.name = "xyx", .value = EulerBasis::XYX},
      {.name = "u", .value = EulerBasis::U},
      {.name = "zsxx", .value = EulerBasis::ZSXX},
  }};
  for (const EulerBasisName& entry : eulerBasisTable) {
    if (basis.equals_insensitive(entry.name)) {
      return entry.value;
    }
  }
  return std::nullopt;
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
