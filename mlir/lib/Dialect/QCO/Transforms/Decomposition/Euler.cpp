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
#include "mlir/Dialect/Utils/Utils.h"

#include <Eigen/Core>
#include <llvm/ADT/STLFunctionalExtras.h>
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
 * @brief Wraps `angle` into `[-pi, pi)`, mapping `+pi` (within `atol`) to
 * `-pi`.
 *
 * @param angle The angle to wrap, in radians.
 * @param atol Absolute tolerance for snapping `+pi` to `-pi`.
 * @return The wrapped angle in `[-pi, pi)`.
 */
[[nodiscard]] static double mod2pi(double angle,
                                   double atol = mlir::utils::TOLERANCE) {
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

  if (wrapped >= pi - atol) {
    wrapped = -pi;
  }

  return wrapped;
}

/**
 * @brief Conjugates a single-qubit matrix by Hadamard (`H * m * H`).
 *
 * Maps X-Y-X / X-Z-X decompositions to Z-Y-Z / Z-X-Z.
 *
 * @param m The single-qubit matrix to conjugate.
 * @return `H * m * H`.
 */
[[nodiscard]] static Eigen::Matrix2cd
hadamardConjugate(const Eigen::Matrix2cd& m) {
  const auto a = m(0, 0);
  const auto b = m(0, 1);
  const auto c = m(1, 0);
  const auto d = m(1, 1);
  return Eigen::Matrix2cd{{0.5 * (a + b + c + d), 0.5 * (a - b + c - d)},
                          {0.5 * (a + b - c - d), 0.5 * (a - b - c + d)}};
}

/**
 * @brief Emits a `GPhaseOp` when `phase` is non-negligible.
 *
 * @param builder Builder used to create the operation.
 * @param loc Source location for the created operation.
 * @param phase Global phase in radians; skipped when within tolerance of zero.
 */
static void emitGPhaseIfNeeded(OpBuilder& builder, Location loc, double phase) {
  if (std::abs(phase) <= mlir::utils::TOLERANCE) {
    return;
  }
  GPhaseOp::create(builder, loc, phase);
}

namespace {

/**
 * @brief Planned RZ-middle-RZ chain; fields are angles in circuit (time) order.
 */
struct PSXSequence {
  enum class Middle : std::uint8_t { OneSX, X, SXRZSX };
  Middle middle = Middle::SXRZSX;
  double firstRZ = 0.0;
  double midRZ = 0.0;
  double lastRZ = 0.0;
};

} // namespace

/**
 * @brief Builds the RZ/SX chain realizing `RZ(phi)*RY(theta)*RZ(lambda)`.
 *
 * Uses the identity `SX*RZ(theta+pi)*SX = Z*RY(theta)`. `theta` from
 * `paramsZYZ` lies in `[0, pi]`: `pi/2` collapses to a single SX; `pi` becomes
 * an X gate (since `SX*SX = X`).
 *
 * @param theta Y-rotation angle in `[0, pi]`.
 * @param phi Trailing Z-rotation angle.
 * @param lambda Leading Z-rotation angle.
 * @return The planned PSX sequence.
 */
[[nodiscard]] static PSXSequence sequenceFromZYZForPSX(double theta, double phi,
                                                       double lambda) {
  constexpr double eps = mlir::utils::TOLERANCE;
  constexpr double halfPi = std::numbers::pi / 2.0;
  constexpr double pi = std::numbers::pi;

  if (std::abs(theta - halfPi) < eps) {
    return {.middle = PSXSequence::Middle::OneSX,
            .firstRZ = lambda - halfPi,
            .midRZ = 0.0,
            .lastRZ = phi + halfPi};
  }
  if (std::abs(theta - pi) < eps) {
    return {.middle = PSXSequence::Middle::X,
            .firstRZ = lambda,
            .midRZ = 0.0,
            .lastRZ = phi + pi};
  }
  return {.middle = PSXSequence::Middle::SXRZSX,
          .firstRZ = lambda,
          .midRZ = theta + pi,
          .lastRZ = phi + pi};
}

/**
 * @brief Global phase between `UOp(theta, phi, lambda)` and the ZYZ product.
 *
 * Relates `UOp` to `RZ(phi)*RY(theta)*RZ(lambda)` on the same angles.
 *
 * @param phi The `phi` Euler angle.
 * @param lambda The `lambda` Euler angle.
 * @return The global-phase offset in radians.
 */
[[nodiscard]] static double globalPhaseOffsetForU(double phi, double lambda) {
  return -0.5 * (phi + lambda);
}

/**
 * @brief Global phase contributed by wrapping an RZ angle with `mod2pi`.
 *
 * `mod2pi(angle) - angle` is a multiple of `2*pi`, so the emitted
 * `RZ(mod2pi(angle))` equals `exp(i*(mod2pi(angle)-angle)/2) * RZ(angle)`.
 *
 * @param angle The unwrapped RZ angle.
 * @return The global-phase contribution in radians.
 */
[[nodiscard]] static double globalPhaseFromRZWrap(double angle) {
  constexpr double eps = mlir::utils::TOLERANCE;
  return 0.5 * (mod2pi(angle, eps) - angle);
}

/**
 * @brief Global phase between the ZYZ product and the emitted PSX product.
 *
 * @param seq The planned PSX sequence.
 * @return The global-phase offset in radians.
 */
[[nodiscard]] static double globalPhaseOffsetForPSX(const PSXSequence& seq) {
  constexpr double halfPi = std::numbers::pi / 2.0;
  constexpr double quarterPi = std::numbers::pi / 4.0;

  switch (seq.middle) {
  case PSXSequence::Middle::OneSX:
    // `SX = exp(i*pi/4)*RZ(-pi/2)*RY(pi/2)*RZ(pi/2)`; the outer RZ angles
    // absorb the +-pi/2, leaving the exp(i*pi/4) phase. RZ wraps add too.
    return -quarterPi + globalPhaseFromRZWrap(seq.firstRZ) +
           globalPhaseFromRZWrap(seq.lastRZ);
  case PSXSequence::Middle::X:
    // `X` swaps the diagonal, so the wraps enter with opposite signs.
    return -halfPi + globalPhaseFromRZWrap(seq.lastRZ) -
           globalPhaseFromRZWrap(seq.firstRZ);
  case PSXSequence::Middle::SXRZSX:
    // `SX*RZ(theta+pi)*SX = Z*RY(theta)`; all three RZ wraps add.
    return halfPi + globalPhaseFromRZWrap(seq.firstRZ) +
           globalPhaseFromRZWrap(seq.midRZ) + globalPhaseFromRZWrap(seq.lastRZ);
  }
  llvm::reportFatalInternalError("Unhandled PSX middle gate");
}

/**
 * @brief Global phase between the ZYZ product and the emitted PSX product.
 *
 * @param theta Y-rotation angle from `paramsZYZ`.
 * @param phi Trailing Z-rotation angle from `paramsZYZ`.
 * @param lambda Leading Z-rotation angle from `paramsZYZ`.
 * @return The global-phase offset in radians.
 */
[[nodiscard]] static double globalPhaseOffsetForPSX(double theta, double phi,
                                                    double lambda) {
  return globalPhaseOffsetForPSX(sequenceFromZYZForPSX(theta, phi, lambda));
}

/**
 * @brief Invokes callbacks for each gate of `seq` in circuit (time) order.
 *
 * @param seq The planned PSX sequence.
 * @param onRZ Called with each RZ angle.
 * @param onSX Called for each SX gate.
 * @param onX Called for each X gate.
 */
static void visitSequenceInTimeOrder(const PSXSequence& seq,
                                     llvm::function_ref<void(double)> onRZ,
                                     llvm::function_ref<void()> onSX,
                                     llvm::function_ref<void()> onX) {
  onRZ(seq.firstRZ);
  switch (seq.middle) {
  case PSXSequence::Middle::OneSX:
    onSX();
    onRZ(seq.lastRZ);
    break;
  case PSXSequence::Middle::X:
    onX();
    onRZ(seq.lastRZ);
    break;
  case PSXSequence::Middle::SXRZSX:
    onSX();
    onRZ(seq.midRZ);
    onSX();
    onRZ(seq.lastRZ);
    break;
  }
}

/**
 * @brief Emits the RZ/SX/X gates of `seq` followed by the global phase.
 *
 * @param builder Builder used to create the operations.
 * @param loc Source location for the created operations.
 * @param qubit Input qubit value.
 * @param seq The planned PSX sequence.
 * @param phase Global phase to emit, in radians.
 * @return The transformed qubit value.
 */
[[nodiscard]] static Value emitFromPSXSequence(OpBuilder& builder, Location loc,
                                               Value qubit,
                                               const PSXSequence& seq,
                                               double phase) {
  constexpr double eps = mlir::utils::TOLERANCE;
  visitSequenceInTimeOrder(
      seq,
      [&](const double angle) {
        qubit =
            RZOp::create(builder, loc, qubit, mod2pi(angle, eps)).getQubitOut();
      },
      [&] { qubit = SXOp::create(builder, loc, qubit).getQubitOut(); },
      [&] { qubit = XOp::create(builder, loc, qubit).getQubitOut(); });
  emitGPhaseIfNeeded(builder, loc, phase);
  return qubit;
}

/**
 * @brief Emits a K-A-K rotation triple plus global phase for `basis`.
 *
 * @param builder Builder used to create the operations.
 * @param loc Source location for the created operations.
 * @param qubit Input qubit value.
 * @param theta Middle (A) rotation angle.
 * @param phi Trailing (K) rotation angle.
 * @param lambda Leading (K) rotation angle.
 * @param phase Global phase to emit, in radians.
 * @param basis Euler basis selecting the K and A rotation axes.
 * @return The transformed qubit value.
 */
static Value emitKAK(OpBuilder& builder, Location loc, Value qubit,
                     double theta, double phi, double lambda, double phase,
                     EulerBasis basis) {
  auto emitK = [&](double a) {
    switch (basis) {
    case EulerBasis::ZYZ:
    case EulerBasis::ZXZ:
      qubit = RZOp::create(builder, loc, qubit, a).getQubitOut();
      break;
    case EulerBasis::XZX:
    case EulerBasis::XYX:
      qubit = RXOp::create(builder, loc, qubit, a).getQubitOut();
      break;
    default:
      llvm::reportFatalInternalError("Invalid K gate for KAK emission");
    }
  };

  auto emitA = [&](double a) {
    switch (basis) {
    case EulerBasis::ZYZ:
    case EulerBasis::XYX:
      qubit = RYOp::create(builder, loc, qubit, a).getQubitOut();
      break;
    case EulerBasis::ZXZ:
      qubit = RXOp::create(builder, loc, qubit, a).getQubitOut();
      break;
    case EulerBasis::XZX:
      qubit = RZOp::create(builder, loc, qubit, a).getQubitOut();
      break;
    default:
      llvm::reportFatalInternalError("Invalid A gate for KAK emission");
    }
  };

  emitK(lambda);
  emitA(theta);
  emitK(phi);
  emitGPhaseIfNeeded(builder, loc, phase);
  return qubit;
}

//===----------------------------------------------------------------------===//
// Euler decomposition (angles)
//===----------------------------------------------------------------------===//

EulerAngles
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
    return paramsU(matrix);
  case EulerBasis::ZSXX:
    return paramsPSX(matrix);
  }
  llvm::reportFatalInternalError(
      "Unsupported Euler basis for angle computation in decomposition!");
}

EulerAngles EulerDecomposition::paramsZYZ(const Eigen::Matrix2cd& matrix) {
  // det(U) = exp(2i*phase); invert the Z-Y-Z parameterization of U's entries.
  const std::complex<double> det =
      matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0);
  const auto detArg = std::arg(det);
  const auto phase = 0.5 * detArg;
  const auto theta =
      2. * std::atan2(std::abs(matrix(1, 0)), std::abs(matrix(0, 0)));
  const auto ang1 = std::arg(matrix(1, 1));
  const auto ang2 = std::arg(matrix(1, 0));
  const auto phi = ang1 + ang2 - detArg;
  const auto lambda = ang1 - ang2;
  return {.theta = theta, .phi = phi, .lambda = lambda, .phase = phase};
}

EulerAngles EulerDecomposition::paramsZXZ(const Eigen::Matrix2cd& matrix) {
  // ZXZ from ZYZ via RY(theta) = RZ(pi/2)*RX(theta)*RZ(-pi/2).
  const auto zyz = paramsZYZ(matrix);
  return {.theta = zyz.theta,
          .phi = zyz.phi + (std::numbers::pi / 2.0),
          .lambda = zyz.lambda - (std::numbers::pi / 2.0),
          .phase = zyz.phase};
}

EulerAngles EulerDecomposition::paramsXYX(const Eigen::Matrix2cd& matrix) {
  // H*RY(theta)*H = RY(-theta): shift outer angles by pi and fix global phase.
  const auto zyz = paramsZYZ(hadamardConjugate(matrix));
  const auto newPhi = mod2pi(zyz.phi + std::numbers::pi, 0.);
  const auto newLambda = mod2pi(zyz.lambda + std::numbers::pi, 0.);
  return {.theta = zyz.theta,
          .phi = newPhi,
          .lambda = newLambda,
          .phase =
              zyz.phase + ((newPhi + newLambda - zyz.phi - zyz.lambda) / 2.)};
}

EulerAngles EulerDecomposition::paramsXZX(const Eigen::Matrix2cd& matrix) {
  // X-Z-X -> Z-X-Z under H conjugation (no Y sign flip, unlike paramsXYX).
  return paramsZXZ(hadamardConjugate(matrix));
}

EulerAngles EulerDecomposition::paramsU(const Eigen::Matrix2cd& matrix) {
  const auto zyz = paramsZYZ(matrix);
  return {.theta = zyz.theta,
          .phi = zyz.phi,
          .lambda = zyz.lambda,
          .phase = zyz.phase + globalPhaseOffsetForU(zyz.phi, zyz.lambda)};
}

EulerAngles EulerDecomposition::paramsPSX(const Eigen::Matrix2cd& matrix) {
  const auto zyz = paramsZYZ(matrix);
  return {.theta = zyz.theta,
          .phi = zyz.phi,
          .lambda = zyz.lambda,
          .phase = zyz.phase +
                   globalPhaseOffsetForPSX(zyz.theta, zyz.phi, zyz.lambda)};
}

//===----------------------------------------------------------------------===//
// Euler synthesis (IR emission)
//===----------------------------------------------------------------------===//

std::optional<EulerBasis> parseEulerBasis(StringRef basis) {
  if (basis.equals_insensitive("zyz")) {
    return EulerBasis::ZYZ;
  }
  if (basis.equals_insensitive("zxz")) {
    return EulerBasis::ZXZ;
  }
  if (basis.equals_insensitive("xzx")) {
    return EulerBasis::XZX;
  }
  if (basis.equals_insensitive("xyx")) {
    return EulerBasis::XYX;
  }
  if (basis.equals_insensitive("u")) {
    return EulerBasis::U;
  }
  if (basis.equals_insensitive("zsxx")) {
    return EulerBasis::ZSXX;
  }
  return std::nullopt;
}

Value synthesizeUnitary1QEuler(OpBuilder& builder, Location loc, Value qubit,
                               const Eigen::Matrix2cd& targetMatrix,
                               EulerBasis basis) {
  if (basis == EulerBasis::ZSXX) {
    const auto zyz = EulerDecomposition::paramsZYZ(targetMatrix);
    const auto seq = sequenceFromZYZForPSX(zyz.theta, zyz.phi, zyz.lambda);
    return emitFromPSXSequence(builder, loc, qubit, seq,
                               zyz.phase + globalPhaseOffsetForPSX(seq));
  }

  const auto angles =
      EulerDecomposition::anglesFromUnitary(targetMatrix, basis);

  switch (basis) {
  case EulerBasis::ZYZ:
  case EulerBasis::ZXZ:
  case EulerBasis::XZX:
  case EulerBasis::XYX:
    qubit = emitKAK(builder, loc, qubit, angles.theta, angles.phi,
                    angles.lambda, angles.phase, basis);
    break;
  case EulerBasis::U:
    qubit = UOp::create(builder, loc, qubit, angles.theta, angles.phi,
                        angles.lambda)
                .getQubitOut();
    emitGPhaseIfNeeded(builder, loc, angles.phase);
    break;
  case EulerBasis::ZSXX:
    llvm_unreachable("ZSXX handled above");
  }

  return qubit;
}

std::size_t synthesisGateCount(const Eigen::Matrix2cd& targetMatrix,
                               EulerBasis basis) {
  switch (basis) {
  case EulerBasis::U:
    return 1;
  case EulerBasis::ZYZ:
  case EulerBasis::ZXZ:
  case EulerBasis::XZX:
  case EulerBasis::XYX:
    // emitKAK always emits the full K-A-K rotation triple.
    return 3;
  case EulerBasis::ZSXX: {
    const auto angles =
        EulerDecomposition::anglesFromUnitary(targetMatrix, EulerBasis::ZSXX);
    const auto seq =
        sequenceFromZYZForPSX(angles.theta, angles.phi, angles.lambda);
    return seq.middle == PSXSequence::Middle::SXRZSX ? 5U : 3U;
  }
  }
  llvm::reportFatalInternalError("Unhandled Euler basis in synthesisGateCount");
}

} // namespace mlir::qco::decomposition
