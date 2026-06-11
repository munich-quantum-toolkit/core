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

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <optional>

namespace mlir::qco::decomposition {

/**
 * @brief Wraps `angle` into `[-pi, pi)`, mapping `+pi` (within `atol`) to
 * `-pi`.
 *
 * @param angle The angle to wrap, in radians.
 * @param atol Tolerance for snapping `+pi` to `-pi`.
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
 * @brief Emits `qco.gphase` when `phase` is outside tolerance.
 *
 * @param builder Builder for the operation.
 * @param loc Location of the operation.
 * @param phase Global phase in radians.
 */
static void emitGPhaseIfNeeded(OpBuilder& builder, Location loc, double phase) {
  if (std::abs(phase) <= mlir::utils::TOLERANCE) {
    return;
  }
  GPhaseOp::create(builder, loc, phase);
}

/**
 * @brief Whether `angle` is numerically zero for gate-emission purposes.
 *
 * @param angle Rotation angle in radians.
 * @return `true` when no rotation gate should be emitted.
 */
[[nodiscard]] static bool isNearZeroRotationAngle(double angle) {
  return std::abs(angle) <= mlir::utils::TOLERANCE;
}

namespace {

/**
 * @brief Planned ZSXX (`RZ` / `SX` / `X`) chain; angles in circuit order.
 */
struct ZSXXSequence {
  ZSXXMiddleGate middle = ZSXXMiddleGate::SXRZSX;
  double firstRZ = 0.0;
  double midRZ = 0.0;
  double lastRZ = 0.0;
};

} // namespace

ZSXXMiddleGate classifyZSXXMiddleFromZYZTheta(double theta) {
  constexpr double eps = mlir::utils::TOLERANCE;
  constexpr double halfPi = std::numbers::pi / 2.0;
  constexpr double pi = std::numbers::pi;

  if (std::abs(theta) <= eps) {
    return ZSXXMiddleGate::OnlyRZ;
  }
  if (std::abs(theta - halfPi) <= eps) {
    return ZSXXMiddleGate::OneSX;
  }
  if (std::abs(theta - pi) <= eps) {
    return ZSXXMiddleGate::X;
  }
  return ZSXXMiddleGate::SXRZSX;
}

/**
 * @brief Builds the ZSXX sequence for `RZ(phi)*RY(theta)*RZ(lambda)`.
 *
 * Uses `SX*RZ(theta+pi)*SX = Z*RY(theta)`.
 *
 * @param theta Y-rotation angle in `[0, pi]`.
 * @param phi Trailing Z-rotation angle.
 * @param lambda Leading Z-rotation angle.
 * @return The planned ZSXX sequence.
 */
[[nodiscard]] static ZSXXSequence
sequenceFromZYZForZSXX(double theta, double phi, double lambda) {
  constexpr double halfPi = std::numbers::pi / 2.0;
  constexpr double pi = std::numbers::pi;

  switch (classifyZSXXMiddleFromZYZTheta(theta)) {
  case ZSXXMiddleGate::OnlyRZ:
    return {.middle = ZSXXMiddleGate::OnlyRZ,
            .firstRZ = lambda,
            .midRZ = 0.0,
            .lastRZ = phi};
  case ZSXXMiddleGate::OneSX:
    return {.middle = ZSXXMiddleGate::OneSX,
            .firstRZ = lambda - halfPi,
            .midRZ = 0.0,
            .lastRZ = phi + halfPi};
  case ZSXXMiddleGate::X:
    return {.middle = ZSXXMiddleGate::X,
            .firstRZ = lambda,
            .midRZ = 0.0,
            .lastRZ = phi + pi};
  case ZSXXMiddleGate::SXRZSX:
    return {.middle = ZSXXMiddleGate::SXRZSX,
            .firstRZ = lambda,
            .midRZ = theta + pi,
            .lastRZ = phi + pi};
  }
  llvm::reportFatalInternalError("Unhandled ZSXX middle gate");
}

/**
 * @brief Global phase offset of `UOp` vs `RZ(phi)*RY(theta)*RZ(lambda)`.
 *
 * @param phi Trailing Z-rotation angle.
 * @param lambda Leading Z-rotation angle.
 * @return The global-phase offset in radians.
 */
[[nodiscard]] static double globalPhaseOffsetForU(double phi, double lambda) {
  return -0.5 * (phi + lambda);
}

/**
 * @brief Global phase from wrapping an RZ angle with `mod2pi`.
 *
 * `RZ(angle + 2*pi) = -RZ(angle)`, so `RZ(mod2pi(angle))` differs from
 * `RZ(angle)` by `exp(i*(mod2pi(angle) - angle)/2)`.
 *
 * @param angle The unwrapped RZ angle.
 * @return The global-phase contribution in radians.
 */
[[nodiscard]] static double globalPhaseFromRZWrap(double angle) {
  constexpr double eps = mlir::utils::TOLERANCE;
  return 0.5 * (mod2pi(angle, eps) - angle);
}

/**
 * @brief Global phase offset of the ZSXX chain vs. the ZYZ product.
 *
 * @param seq The planned ZSXX sequence.
 * @return The global-phase offset in radians.
 */
[[nodiscard]] static double globalPhaseOffsetForZSXX(const ZSXXSequence& seq) {
  constexpr double halfPi = std::numbers::pi / 2.0;
  constexpr double quarterPi = std::numbers::pi / 4.0;

  switch (seq.middle) {
  case ZSXXMiddleGate::OnlyRZ:
    return globalPhaseFromRZWrap(seq.firstRZ) +
           globalPhaseFromRZWrap(seq.lastRZ);
  case ZSXXMiddleGate::OneSX:
    // `SX = exp(i*pi/4)*RZ(-pi/2)*RY(pi/2)*RZ(pi/2)`; the outer RZ angles
    // absorb the +-pi/2, leaving the exp(i*pi/4) phase. RZ wraps add too.
    return -quarterPi + globalPhaseFromRZWrap(seq.firstRZ) +
           globalPhaseFromRZWrap(seq.lastRZ);
  case ZSXXMiddleGate::X:
    // `X` swaps the diagonal, so the wraps enter with opposite signs.
    return -halfPi + globalPhaseFromRZWrap(seq.lastRZ) -
           globalPhaseFromRZWrap(seq.firstRZ);
  case ZSXXMiddleGate::SXRZSX:
    // `SX*RZ(theta+pi)*SX = Z*RY(theta)`; all three RZ wraps add.
    return halfPi + globalPhaseFromRZWrap(seq.firstRZ) +
           globalPhaseFromRZWrap(seq.midRZ) + globalPhaseFromRZWrap(seq.lastRZ);
  }
  llvm::reportFatalInternalError("Unhandled ZSXX middle gate");
}

/**
 * @brief Invokes callbacks for each gate of `seq` in circuit order.
 *
 * @param seq The planned ZSXX sequence.
 * @param onRZ Called with each RZ angle.
 * @param onSX Called for each SX gate.
 * @param onX Called for each X gate.
 */
static void visitSequenceInTimeOrder(const ZSXXSequence& seq,
                                     llvm::function_ref<void(double)> onRZ,
                                     llvm::function_ref<void()> onSX,
                                     llvm::function_ref<void()> onX) {
  onRZ(seq.firstRZ);
  switch (seq.middle) {
  case ZSXXMiddleGate::OnlyRZ:
    onRZ(seq.lastRZ);
    break;
  case ZSXXMiddleGate::OneSX:
    onSX();
    onRZ(seq.lastRZ);
    break;
  case ZSXXMiddleGate::X:
    onX();
    onRZ(seq.lastRZ);
    break;
  case ZSXXMiddleGate::SXRZSX:
    onSX();
    onRZ(seq.midRZ);
    onSX();
    onRZ(seq.lastRZ);
    break;
  }
}

/**
 * @brief Emits the gates of `seq` and optional `gphase`.
 *
 * @param builder Builder for the operations.
 * @param loc Location of the operations.
 * @param qubit Input qubit value.
 * @param seq The planned ZSXX sequence.
 * @param phase Global phase in radians.
 * @return The output qubit value.
 */
[[nodiscard]] static Value emitFromZSXXSequence(OpBuilder& builder,
                                                Location loc, Value qubit,
                                                const ZSXXSequence& seq,
                                                double phase) {
  constexpr double eps = mlir::utils::TOLERANCE;
  visitSequenceInTimeOrder(
      seq,
      [&](const double angle) {
        const double wrapped = mod2pi(angle, eps);
        if (isNearZeroRotationAngle(wrapped)) {
          return;
        }
        qubit = RZOp::create(builder, loc, qubit, wrapped).getQubitOut();
      },
      [&] { qubit = SXOp::create(builder, loc, qubit).getQubitOut(); },
      [&] { qubit = XOp::create(builder, loc, qubit).getQubitOut(); });
  emitGPhaseIfNeeded(builder, loc, phase);
  return qubit;
}

/**
 * @brief Emits a K-A-K rotation triple and optional `gphase` for `basis`.
 *
 * @param builder Builder for the operations.
 * @param loc Location of the operations.
 * @param qubit Input qubit value.
 * @param theta Middle (A) rotation angle.
 * @param phi Trailing (K) rotation angle.
 * @param lambda Leading (K) rotation angle.
 * @param phase Global phase in radians.
 * @param basis Euler basis selecting the rotation axes.
 * @return The output qubit value.
 */
static Value emitKAK(OpBuilder& builder, Location loc, Value qubit,
                     double theta, double phi, double lambda, double phase,
                     EulerBasis basis) {
  auto emitK = [&](double a) {
    if (isNearZeroRotationAngle(a)) {
      return;
    }
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
    if (isNearZeroRotationAngle(a)) {
      return;
    }
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

EulerAngles EulerDecomposition::anglesFromUnitary(const Matrix2x2& matrix,
                                                  EulerBasis basis) {
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
  }
  llvm::reportFatalInternalError(
      "Unsupported Euler basis for angle computation in decomposition!");
}

EulerAngles EulerDecomposition::paramsZYZ(const Matrix2x2& matrix) {
  // det(U) = exp(2i*phase); invert the Z-Y-Z parameterization of U's entries.
  const std::complex<double> det =
      matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0);
  const auto detArg = std::arg(det);
  const auto phase = 0.5 * detArg;
  const auto theta =
      2. * std::atan2(std::abs(matrix(1, 0)), std::abs(matrix(0, 0)));
  const auto ang1 = std::arg(matrix(1, 1));
  constexpr double eps = mlir::utils::TOLERANCE;
  double ang2 = 0.0;
  if (std::abs(matrix(1, 0)) > eps) {
    ang2 = std::arg(matrix(1, 0));
  } else if (std::abs(matrix(0, 1)) > eps) {
    ang2 = std::arg(matrix(0, 1));
  }
  const auto phi = ang1 + ang2 - detArg;
  const auto lambda = ang1 - ang2;
  return {.theta = theta, .phi = phi, .lambda = lambda, .phase = phase};
}

EulerAngles EulerDecomposition::paramsZXZ(const Matrix2x2& matrix) {
  // ZXZ from ZYZ via RY(theta) = RZ(pi/2)*RX(theta)*RZ(-pi/2).
  const auto zyz = paramsZYZ(matrix);
  return {.theta = zyz.theta,
          .phi = zyz.phi + (std::numbers::pi / 2.0),
          .lambda = zyz.lambda - (std::numbers::pi / 2.0),
          .phase = zyz.phase};
}

EulerAngles EulerDecomposition::paramsXZX(const Matrix2x2& matrix) {
  // X-Z-X -> Z-X-Z under H conjugation (no Y sign flip, unlike paramsXYX).
  return paramsZXZ(hadamardConjugate(matrix));
}

EulerAngles EulerDecomposition::paramsXYX(const Matrix2x2& matrix) {
  // H*RY(theta)*H = RY(-theta): shift outer angles by pi and fix global phase.
  const auto zyz = paramsZYZ(hadamardConjugate(matrix));
  // Keep atol=0 so `phase` tracks the unwrapped ZYZ angles; snapping to pi
  // would change the recorded global-phase correction.
  const auto newPhi = mod2pi(zyz.phi + std::numbers::pi, 0.);
  const auto newLambda = mod2pi(zyz.lambda + std::numbers::pi, 0.);
  return {.theta = zyz.theta,
          .phi = newPhi,
          .lambda = newLambda,
          .phase =
              zyz.phase + ((newPhi + newLambda - zyz.phi - zyz.lambda) / 2.)};
}

EulerAngles EulerDecomposition::paramsU(const Matrix2x2& matrix) {
  const auto zyz = paramsZYZ(matrix);
  return {.theta = zyz.theta,
          .phi = zyz.phi,
          .lambda = zyz.lambda,
          .phase = zyz.phase + globalPhaseOffsetForU(zyz.phi, zyz.lambda)};
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
                               const Matrix2x2& targetMatrix,
                               EulerBasis basis) {
  if (basis == EulerBasis::ZSXX) {
    const auto zyz = EulerDecomposition::paramsZYZ(targetMatrix);
    const auto seq = sequenceFromZYZForZSXX(zyz.theta, zyz.phi, zyz.lambda);
    return emitFromZSXXSequence(builder, loc, qubit, seq,
                                zyz.phase + globalPhaseOffsetForZSXX(seq));
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

/**
 * @brief Counts non-identity K-A-K rotations `emitKAK` would emit.
 */
[[nodiscard]] static std::size_t countKAKGates(double theta, double phi,
                                               double lambda) {
  std::size_t count = 0;
  if (!isNearZeroRotationAngle(lambda)) {
    ++count;
  }
  if (!isNearZeroRotationAngle(theta)) {
    ++count;
  }
  if (!isNearZeroRotationAngle(phi)) {
    ++count;
  }
  return count;
}

/**
 * @brief Counts non-zero `RZ` slots in a ZSXX sequence angle.
 */
[[nodiscard]] static std::size_t countNonZeroZSXXAngle(double angle) {
  constexpr double eps = mlir::utils::TOLERANCE;
  return isNearZeroRotationAngle(mod2pi(angle, eps)) ? 0 : 1;
}

/**
 * @brief Counts basis gates `emitFromZSXXSequence` would emit for `seq`.
 */
[[nodiscard]] static std::size_t
countZSXXSequenceGates(const ZSXXSequence& seq) {
  switch (seq.middle) {
  case ZSXXMiddleGate::OnlyRZ:
    return countNonZeroZSXXAngle(seq.firstRZ) +
           countNonZeroZSXXAngle(seq.lastRZ);
  case ZSXXMiddleGate::OneSX:
  case ZSXXMiddleGate::X:
    return countNonZeroZSXXAngle(seq.firstRZ) + 1 +
           countNonZeroZSXXAngle(seq.lastRZ);
  case ZSXXMiddleGate::SXRZSX:
    return countNonZeroZSXXAngle(seq.firstRZ) + 1 +
           countNonZeroZSXXAngle(seq.midRZ) + 1 +
           countNonZeroZSXXAngle(seq.lastRZ);
  }
  llvm::reportFatalInternalError("Unhandled ZSXX middle gate in gate count");
}

std::size_t synthesisGateCount(const Matrix2x2& targetMatrix,
                               EulerBasis basis) {
  if (basis == EulerBasis::U) {
    return 1;
  }

  if (targetMatrix.isApprox(Matrix2x2::identity())) {
    return 0;
  }

  switch (basis) {
  case EulerBasis::ZYZ:
  case EulerBasis::ZXZ:
  case EulerBasis::XZX:
  case EulerBasis::XYX: {
    const auto angles =
        EulerDecomposition::anglesFromUnitary(targetMatrix, basis);
    return countKAKGates(angles.theta, angles.phi, angles.lambda);
  }
  case EulerBasis::ZSXX: {
    const auto zyz =
        EulerDecomposition::anglesFromUnitary(targetMatrix, EulerBasis::ZYZ);
    const auto seq = sequenceFromZYZForZSXX(zyz.theta, zyz.phi, zyz.lambda);
    return countZSXXSequenceGates(seq);
  }
  }
  llvm::reportFatalInternalError("Unhandled Euler basis in synthesisGateCount");
}

bool wouldShortenInBasisRun(const std::size_t runSize,
                            const Matrix2x2& composed, EulerBasis basis) {
  if (runSize > maxSynthesisGateCount(basis)) {
    return true;
  }
  return runSize > synthesisGateCount(composed, basis);
}

} // namespace mlir::qco::decomposition
