/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Transforms/Decomposition/Euler.h"

#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Transforms/Decomposition/Pauli.h"
#include "mqt/Dialect/QCO/Utils/Matrix.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <utility>
#include <variant>

namespace mlir::qco::decomposition {

bool isSingleQubitBasisGate(Operation* op, SingleQubitBasis basis) {
  return TypeSwitch<Operation*, bool>(op)
      .Case([&](RZOp) {
        return basis == SingleQubitBasis::ZYZ ||
               basis == SingleQubitBasis::ZXZ ||
               basis == SingleQubitBasis::XZX ||
               basis == SingleQubitBasis::ZSXX;
      })
      .Case([&](RYOp) {
        return basis == SingleQubitBasis::ZYZ || basis == SingleQubitBasis::XYX;
      })
      .Case([&](RXOp) {
        return basis == SingleQubitBasis::ZXZ ||
               basis == SingleQubitBasis::XZX || basis == SingleQubitBasis::XYX;
      })
      .Case([&](UOp) { return basis == SingleQubitBasis::U; })
      .Case<SXOp, SXdgOp, XOp>(
          [&](auto) { return basis == SingleQubitBasis::ZSXX; })
      .Case([&](ROp) { return basis == SingleQubitBasis::R; })
      .Default([](auto) { return false; });
}

/// Wraps `angle` into `[-π, π)`, mapping `+π` (within tolerance) to `-π`.
///
/// @param angle The angle to wrap, in radians.
/// @return The wrapped angle in `[-π, π)`.
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

  if (wrapped >= pi - mqt::PARAMETER_COMPARISON_TOLERANCE) {
    wrapped = -pi;
  }

  return wrapped;
}

/// Conjugates a single-qubit matrix by Hadamard (`H * m * H`).
///
/// Maps XYX / XZX parameterizations to ZYZ / ZXZ.
///
/// @param m The single-qubit matrix to conjugate.
/// @return `H * m * H`.
[[nodiscard]] static Matrix2x2 hadamardConjugate(const Matrix2x2& m) {
  const auto a = m(0, 0);
  const auto b = m(0, 1);
  const auto c = m(1, 0);
  const auto d = m(1, 1);
  return Matrix2x2::fromElements(0.5 * (a + b + c + d), 0.5 * (a - b + c - d),
                                 0.5 * (a + b - c - d), 0.5 * (a - b - c + d));
}

/// Whether `angle` is numerically zero for gate-emission purposes.
///
/// Use matrix precision here; fixed target capabilities remain exact.
///
/// @param angle Rotation angle in radians.
/// @return `true` when no rotation gate should be emitted.
[[nodiscard]] static bool isNearZeroRotationAngle(const double angle) {
  return std::abs(angle) <= MATRIX_TOLERANCE;
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

/// Z-Y-Z Euler angles and global phase for a 2x2 unitary.
///
/// @param matrix Single-qubit unitary to decompose.
/// @return Z-Y-Z angles and global phase.
[[nodiscard]] static EulerAngles paramsZYZ(const Matrix2x2& matrix) {
  // det(U) = exp(2i*phase)
  const Complex det = matrix.determinant();
  const auto detArg = std::arg(det);
  const auto phase = 0.5 * detArg;
  const auto theta =
      2. * std::atan2(std::abs(matrix(1, 0)), std::abs(matrix(0, 0)));
  const auto ang1 = std::arg(matrix(1, 1));
  double ang2 = 0.0;
  if (std::abs(matrix(1, 0)) > mqt::PARAMETER_COMPARISON_TOLERANCE) {
    ang2 = std::arg(matrix(1, 0));
  } else if (std::abs(matrix(0, 1)) > mqt::PARAMETER_COMPARISON_TOLERANCE) {
    ang2 = std::arg(matrix(0, 1));
  }
  const auto phi = ang1 + ang2 - detArg;
  const auto lambda = ang1 - ang2;
  return {.theta = theta, .phi = phi, .lambda = lambda, .phase = phase};
}

/// Z-X-Z Euler angles via `RY(θ) = RZ(π/2)*RX(θ)*RZ(-π/2)`.
///
/// @param matrix Single-qubit unitary to decompose.
/// @return Z-X-Z angles and global phase.
[[nodiscard]] static EulerAngles paramsZXZ(const Matrix2x2& matrix) {
  const auto [theta, phi, lambda, phase] = paramsZYZ(matrix);
  return {
      .theta = theta,
      .phi = phi + (std::numbers::pi / 2.0),
      .lambda = lambda - (std::numbers::pi / 2.0),
      .phase = phase,
  };
}

/// X-Z-X Euler angles (Z-X-Z under H conjugation).
///
/// @param matrix Single-qubit unitary to decompose.
/// @return X-Z-X angles and global phase.
[[nodiscard]] static EulerAngles paramsXZX(const Matrix2x2& matrix) {
  return paramsZXZ(hadamardConjugate(matrix));
}

/// X-Y-X Euler angles via `H*RY(θ)*H = RY(-θ)`.
///
/// @param matrix Single-qubit unitary to decompose.
/// @return X-Y-X angles and global phase.
[[nodiscard]] static EulerAngles paramsXYX(const Matrix2x2& matrix) {
  // Shift outer angles by π and fix global phase.
  const auto [theta, phi, lambda, phase] = paramsZYZ(hadamardConjugate(matrix));
  return {
      .theta = theta,
      .phi = phi + std::numbers::pi,
      .lambda = lambda + std::numbers::pi,
      .phase = phase + std::numbers::pi,
  };
}

/// `U`-basis angles (Z-Y-Z angles with a `U`-vs-`RZ*RY*RZ` phase fix).
///
/// @param matrix Single-qubit unitary to decompose.
/// @return `U`-gate angles and global phase.
[[nodiscard]] static EulerAngles paramsU(const Matrix2x2& matrix) {
  // `U` differs from RZ(φ)*RY(θ)*RZ(λ) by a global phase of
  // -(φ + λ)/2.
  const auto [theta, phi, lambda, phase] = paramsZYZ(matrix);
  return {
      .theta = theta,
      .phi = phi,
      .lambda = lambda,
      .phase = phase - (0.5 * (phi + lambda)),
  };
}

EulerAngles anglesFromUnitary(const Matrix2x2& matrix,
                              const SingleQubitBasis basis) {
  switch (basis) {
  case SingleQubitBasis::ZYZ:
  case SingleQubitBasis::ZSXX:
  case SingleQubitBasis::R:
    return paramsZYZ(matrix);
  case SingleQubitBasis::ZXZ:
    return paramsZXZ(matrix);
  case SingleQubitBasis::XZX:
    return paramsXZX(matrix);
  case SingleQubitBasis::XYX:
    return paramsXYX(matrix);
  case SingleQubitBasis::U:
    return paramsU(matrix);
  default:
    llvm_unreachable("invalid single-qubit synthesis basis");
  }
}

//===----------------------------------------------------------------------===//
// Euler synthesis (plan + emit)
//===----------------------------------------------------------------------===//

namespace {

struct SynthesisStep {
  enum class Kind : uint8_t { RZ, RY, RX, SX, SXdg, X, U, R };
  Kind kind;
  RotationParameter theta = 0.;
  RotationParameter phi = 0.;
  RotationParameter lambda = 0.;
};

struct Unitary1QEulerPlan {
  SmallVector<SynthesisStep, 5> steps;
  RotationParameter phase = 0.;
};

} // namespace

RotationParameter normalizeRotationParameter(OpBuilder& builder, Location loc,
                                             RotationParameter angle) {
  if (const auto value = mqt::parameterToConstantDouble(angle)) {
    return std::abs(*value) <= 2. * std::numbers::pi
               ? *value
               : 4. * std::atan(std::tan(*value / 4.));
  }
  const auto four = mqt::FloatExpression::constant(builder, loc, 4.);
  const mqt::FloatExpression value(builder, loc, std::get<Value>(angle));
  return ((value / four).tan().atan() * four).getValue();
}

void emitGPhaseIfNeeded(OpBuilder& builder, Location loc, Value phase) {
  const auto normalized = normalizeRotationParameter(builder, loc, phase);
  if (const auto constant = mqt::parameterToConstantDouble(normalized)) {
    emitGPhaseIfNeeded(builder, loc, *constant);
  } else {
    GPhaseOp::create(builder, loc, std::get<Value>(normalized));
  }
}

void simplifyRotationAngles(SmallVectorImpl<RotationAngleTerm>& angles) {
  const auto dyadic = [](double scale) {
    int exponent = 0;
    return std::abs(scale) <= 1. &&
           std::frexp(std::abs(scale), &exponent) == 0.5;
  };
  // Keep per-key storage compact.
  DenseMap<std::pair<Value, uint64_t>, SmallVector<size_t, 1>> unmatched;
  for (auto [index, term] : llvm::enumerate(angles)) {
    if (term.scale == 0.) {
      continue;
    }
    while (true) {
      Value operand;
      double scale = 0.;
      if (auto negative = term.value.getDefiningOp<arith::NegFOp>()) {
        operand = negative.getOperand();
        scale = -1.;
      } else if (auto product = term.value.getDefiningOp<arith::MulFOp>()) {
        if (auto factor = mqt::valueToConstantDouble(product.getRhs())) {
          operand = product.getLhs();
          scale = *factor;
        } else if (auto factor = mqt::valueToConstantDouble(product.getLhs())) {
          operand = product.getRhs();
          scale = *factor;
        }
      } else if (auto quotient = term.value.getDefiningOp<arith::DivFOp>()) {
        if (auto divisor = mqt::valueToConstantDouble(quotient.getRhs())) {
          operand = quotient.getLhs();
          scale = 1. / *divisor;
        }
      }
      if (!operand || !dyadic(scale) || term.scale * scale == 0.) {
        break;
      }
      term.value = operand;
      term.scale *= scale;
    }
    if (mqt::valueToConstantDouble(term.value) == 0.) {
      term.scale = 0.;
    }
    if (term.scale == 0.) {
      continue;
    }
    auto& indices =
        unmatched[{term.value, std::bit_cast<uint64_t>(std::abs(term.scale))}];
    if (!indices.empty() && std::signbit(angles[indices.back()].scale) !=
                                std::signbit(term.scale)) {
      angles[indices.pop_back_val()].scale = 0.;
      term.scale = 0.;
    } else {
      indices.push_back(index);
    }
  }
  llvm::erase_if(angles, [](const auto& term) { return term.scale == 0.; });
}

Value emitRotationAngleSum(OpBuilder& builder, Location loc,
                           ArrayRef<RotationAngleTerm> terms) {
  if (terms.empty()) {
    return mqt::constantFromScalar(builder, loc, 0.);
  }
  SmallVector<Value> angles;
  angles.reserve(terms.size());
  for (const auto& term : terms) {
    Value angle = term.scale == 1.
                      ? term.value
                      : builder.createOrFold<arith::MulFOp>(
                            loc, term.value,
                            mqt::constantFromScalar(builder, loc, term.scale));
    angles.push_back(terms.size() == 1
                         ? angle
                         : mqt::variantToValue(builder, loc,
                                               normalizeRotationParameter(
                                                   builder, loc, angle)));
  }
  for (size_t stride = 1; stride < angles.size(); stride *= 2) {
    for (size_t i = 0; i + stride < angles.size(); i += 2 * stride) {
      angles[i] = builder.createOrFold<arith::AddFOp>(loc, angles[i],
                                                      angles[i + stride]);
    }
  }
  return angles.front();
}

Value sumRotationAngles(OpBuilder& builder, Location loc,
                        ArrayRef<Value> inputs) {
  SmallVector<RotationAngleTerm> angles;
  angles.reserve(inputs.size());
  for (auto input : inputs) {
    angles.push_back({.value = input});
  }
  simplifyRotationAngles(angles);
  return emitRotationAngleSum(builder, loc, angles);
}

static bool isConstantParameter(const RotationParameter& value,
                                double expected = 0.) {
  const auto scalar = mqt::parameterToConstantDouble(value);
  return scalar && isNearZeroRotationAngle(*scalar - expected);
}

/// A finite rotation angle has an exact additive identity.
///
/// Keep this policy outside scalar arithmetic, where signed zero and nonfinite
/// values matter.
static RotationParameter addRotationParameters(OpBuilder& builder, Location loc,
                                               const RotationParameter& lhs,
                                               const RotationParameter& rhs) {
  const auto foldedLhs = mqt::foldParameter(lhs);
  const auto foldedRhs = mqt::foldParameter(rhs);
  const auto* a = std::get_if<double>(&foldedLhs);
  const auto* b = std::get_if<double>(&foldedRhs);
  if (a != nullptr && b != nullptr) {
    return *a + *b;
  }
  if (a != nullptr && *a == 0.) {
    return rhs;
  }
  if (b != nullptr && *b == 0.) {
    return lhs;
  }
  return (mqt::FloatExpression(builder, loc, foldedLhs) +
          mqt::FloatExpression(builder, loc, foldedRhs))
      .getValue();
}

/// One emission recipe serves extracted numeric angles and known SSA angles.
static Unitary1QEulerPlan
planEulerAngles(OpBuilder& builder, Location loc,
                const std::array<RotationParameter, 4>& angles,
                const CompilerTarget::SynthesisBasis& basis) {
  const auto& [theta, phi, lambda, phase] = angles;
  Unitary1QEulerPlan plan{.phase = phase};
  const auto add = [&](const RotationParameter& a, const RotationParameter& b) {
    return addRotationParameters(builder, loc, a, b);
  };
  const auto rotation = [&](SynthesisStep::Kind kind,
                            const RotationParameter& angle,
                            const RotationParameter& axis = 0.) {
    RotationParameter normalized = angle;
    if (const auto value = mqt::parameterToConstantDouble(angle)) {
      const double wrapped = mod2pi(*value);
      // Removing a full Pauli turn contributes a minus sign.
      plan.phase = add(plan.phase, 0.5 * (*value - wrapped));
      normalized = wrapped;
    }
    if (!isConstantParameter(normalized)) {
      plan.steps.push_back({
          .kind = kind,
          .theta = normalized,
          .phi = axis,
      });
    }
  };
  using Kind = SynthesisStep::Kind;
  if (basis.singleQubit == SingleQubitBasis::R) {
    constexpr double pi = std::numbers::pi;
    const auto sum = mqt::parameterToConstantDouble(add(phi, lambda));
    if (sum && std::abs(std::sin(*sum / 2.)) < MATRIX_TOLERANCE) {
      plan.phase = add(plan.phase, *sum / 2.);
      rotation(Kind::R, theta, add(phi, pi / 2.));
    } else {
      // RZ(phi) RY(theta) RZ(lambda) is a product of two equatorial rotations.
      const auto negativeLambda =
          mqt::scaleParameter(builder, loc, lambda, -1.);
      const auto axis = mqt::scaleParameter(
          builder, loc, add(add(phi, negativeLambda), pi), 0.5);
      rotation(Kind::R,
               add(normalizeRotationParameter(builder, loc, theta), -pi),
               add(negativeLambda, pi / 2.));
      rotation(Kind::R, pi, axis);
    }
    return plan;
  }
  if (isConstantParameter(theta)) {
    switch (basis.singleQubit) {
    case SingleQubitBasis::ZYZ:
    case SingleQubitBasis::ZXZ:
    case SingleQubitBasis::ZSXX:
      rotation(Kind::RZ, add(phi, lambda));
      break;
    case SingleQubitBasis::XZX:
    case SingleQubitBasis::XYX:
      rotation(Kind::RX, add(phi, lambda));
      break;
    case SingleQubitBasis::R:
      llvm_unreachable("R synthesis handled above");
    case SingleQubitBasis::U:
      if (const auto p = mqt::parameterToConstantDouble(phi),
          l = mqt::parameterToConstantDouble(lambda);
          !p || !l || !isNearZeroRotationAngle(mod2pi(*p + *l))) {
        plan.steps.push_back(
            {.kind = Kind::U, .theta = 0., .phi = phi, .lambda = lambda});
      }
      break;
    }
    return plan;
  }
  switch (basis.singleQubit) {
  case SingleQubitBasis::ZYZ:
  case SingleQubitBasis::ZXZ:
    rotation(Kind::RZ, lambda);
    rotation(basis.singleQubit == SingleQubitBasis::ZYZ ? Kind::RY : Kind::RX,
             theta);
    rotation(Kind::RZ, phi);
    break;
  case SingleQubitBasis::XZX:
  case SingleQubitBasis::XYX:
    rotation(Kind::RX, lambda);
    rotation(basis.singleQubit == SingleQubitBasis::XZX ? Kind::RZ : Kind::RY,
             theta);
    rotation(Kind::RX, phi);
    break;
  case SingleQubitBasis::R:
    llvm_unreachable("R synthesis handled above");
  case SingleQubitBasis::U:
    plan.steps.push_back(
        {.kind = Kind::U, .theta = theta, .phi = phi, .lambda = lambda});
    break;
  case SingleQubitBasis::ZSXX: {
    constexpr double pi = std::numbers::pi;
    constexpr double halfPi = pi / 2.;
    const bool inverse =
        basis.quarterTurnGates && basis.quarterTurnGates->quarterTurnAngle < 0.;
    // RY(t) = RZ(pi/2) RX(t) RZ(-pi/2); absorb this frame in the outer RZs.
    const double azimuth =
        basis.quarterTurnGates &&
                basis.quarterTurnGates->gate == CompilerTarget::GateKind::RY
            ? halfPi
            : 0.;
    const double offset = (inverse ? pi : 0.) + azimuth;
    const double quarterPhase = inverse ? -pi / 4. : pi / 4.;
    const auto quarterTurn = inverse ? Kind::SXdg : Kind::SX;
    if (isConstantParameter(theta, halfPi)) {
      rotation(Kind::RZ, add(lambda, offset - halfPi));
      plan.steps.push_back({.kind = quarterTurn});
      rotation(Kind::RZ, add(phi, halfPi - offset));
      plan.phase = add(plan.phase, -quarterPhase);
      break;
    }
    if (basis.hasHalfTurn && isConstantParameter(theta, pi)) {
      // A half turn reverses the Z axis, so the outer rotations combine.
      plan.steps.push_back({.kind = Kind::X});
      rotation(Kind::RZ,
               add(add(phi, mqt::scaleParameter(builder, loc, lambda, -1.)),
                   pi - 2. * azimuth));
      plan.phase = add(plan.phase, -halfPi);
      break;
    }
    rotation(Kind::RZ, add(lambda, offset));
    plan.steps.push_back({.kind = quarterTurn});
    rotation(Kind::RZ, add(theta, pi));
    plan.steps.push_back({.kind = quarterTurn});
    rotation(Kind::RZ, add(phi, pi - offset));
    plan.phase = add(plan.phase, pi - 2. * quarterPhase);
    break;
  }
  }
  return plan;
}

/// Materialize the selected target operations while retaining the phase.
static std::pair<Value, RotationParameter>
emitEulerPlan(OpBuilder& builder, Location loc, Value qubit,
              const Unitary1QEulerPlan& plan,
              const CompilerTarget::SynthesisBasis& basis) {
  auto phase = plan.phase;
  const auto parameter = [&](const RotationParameter& value) {
    return mqt::variantToValue(builder, loc, value);
  };
  for (const auto& [kind, theta, phi, lambda] : plan.steps) {
    using Kind = SynthesisStep::Kind;
    std::optional<double> nativeAngle;
    if (basis.quarterTurnGates) {
      if (kind == Kind::SX || kind == Kind::SXdg) {
        nativeAngle = basis.quarterTurnGates->quarterTurnAngle;
      } else if (kind == Kind::X) {
        nativeAngle = basis.quarterTurnGates->halfTurnAngle;
      }
    }
    if (nativeAngle) {
      qubit =
          basis.quarterTurnGates->gate == CompilerTarget::GateKind::R
              ? ROp::create(builder, loc, qubit, *nativeAngle, 0.).getQubitOut()
          : basis.quarterTurnGates->gate == CompilerTarget::GateKind::RY
              ? RYOp::create(builder, loc, qubit, *nativeAngle).getQubitOut()
              : RXOp::create(builder, loc, qubit, *nativeAngle).getQubitOut();
      phase = addRotationParameters(builder, loc, phase, *nativeAngle / 2.);
      continue;
    }
    switch (kind) {
    case Kind::RZ:
      qubit = RZOp::create(builder, loc, qubit, parameter(theta)).getQubitOut();
      break;
    case Kind::RY:
      qubit = RYOp::create(builder, loc, qubit, parameter(theta)).getQubitOut();
      break;
    case Kind::RX:
      qubit = RXOp::create(builder, loc, qubit, parameter(theta)).getQubitOut();
      break;
    case Kind::SX:
      qubit = SXOp::create(builder, loc, qubit).getQubitOut();
      break;
    case Kind::SXdg:
      qubit = SXdgOp::create(builder, loc, qubit).getQubitOut();
      break;
    case Kind::X:
      qubit = XOp::create(builder, loc, qubit).getQubitOut();
      break;
    case Kind::U:
      qubit = UOp::create(builder, loc, qubit, parameter(theta), parameter(phi),
                          parameter(lambda))
                  .getQubitOut();
      break;
    case Kind::R:
      qubit = ROp::create(builder, loc, qubit, parameter(theta), parameter(phi))
                  .getQubitOut();
      break;
    }
  }
  return {qubit, phase};
}

static void emitParameterPhase(OpBuilder& builder, Location loc,
                               const RotationParameter& phase) {
  if (!isConstantParameter(phase)) {
    GPhaseOp::create(builder, loc, mqt::variantToValue(builder, loc, phase));
  }
}

std::optional<SingleQubitBasis> parseSingleQubitBasis(StringRef basis) {
  return StringSwitch<std::optional<SingleQubitBasis>>(basis.lower())
      .Case("zyz", SingleQubitBasis::ZYZ)
      .Case("zxz", SingleQubitBasis::ZXZ)
      .Case("xzx", SingleQubitBasis::XZX)
      .Case("xyx", SingleQubitBasis::XYX)
      .Case("u", SingleQubitBasis::U)
      .Case("zsxx", SingleQubitBasis::ZSXX)
      .Case("r", SingleQubitBasis::R)
      .Default(std::nullopt);
}

std::optional<SynthesizedUnitary1Q>
synthesizeUnitary1QEuler(OpBuilder& builder, Location loc, Value qubit,
                         const Matrix2x2& composed, size_t runSize,
                         bool hasNonBasisGate,
                         const CompilerTarget::SynthesisBasis& basis) {
  Unitary1QEulerPlan plan;
  if (!composed.isApprox(Matrix2x2::identity())) {
    const auto angles = anglesFromUnitary(composed, basis.singleQubit);
    plan = planEulerAngles(builder, loc,
                           {
                               angles.theta,
                               angles.phi,
                               angles.lambda,
                               angles.phase,
                           },
                           basis);
    // K(phi) A(theta) K(lambda) = K(phi+pi) A(-theta) K(lambda-pi).
    // Keep the alternate Euler representative only when it removes gates.
    constexpr double pi = std::numbers::pi;
    if (basis.singleQubit != SingleQubitBasis::U && plan.steps.size() > 1 &&
        (isNearZeroRotationAngle(mod2pi(angles.phi + pi)) ||
         isNearZeroRotationAngle(mod2pi(angles.lambda + pi)))) {
      auto alternate = planEulerAngles(
          builder, loc,
          {-angles.theta, angles.phi + pi, angles.lambda - pi, angles.phase},
          basis);
      if (alternate.steps.size() < plan.steps.size()) {
        plan = std::move(alternate);
      }
    }
  }
  if (!hasNonBasisGate && runSize <= plan.steps.size()) {
    return std::nullopt;
  }
  auto [output, phase] = emitEulerPlan(builder, loc, qubit, plan, basis);
  return SynthesizedUnitary1Q{
      .qubit = output,
      .globalPhase = std::get<double>(phase),
  };
}

Value emitParameterizedEulerAngles(
    OpBuilder& builder, Location loc, Value qubit,
    const std::array<RotationParameter, 4>& angles,
    const CompilerTarget::SynthesisBasis& basis) {
  const auto plan = planEulerAngles(builder, loc, angles, basis);
  auto [output, phase] = emitEulerPlan(builder, loc, qubit, plan, basis);
  emitParameterPhase(builder, loc, phase);
  return output;
}

Value synthesizePauliRotation1Q(OpBuilder& builder, Location loc, Value qubit,
                                PauliAxis axis, RotationParameter angle,
                                const CompilerTarget::SynthesisBasis& basis) {
  if (axis == PauliAxis::I) {
    llvm_unreachable("single-qubit synthesis requires a nonidentity Pauli");
  }
  if (const auto constant = mqt::parameterToConstantDouble(angle)) {
    const auto frame = pauliFrame(axis);
    const auto matrix =
        frame * RZOp::unitaryMatrix(*constant) * frame.adjoint();
    const auto result =
        synthesizeUnitary1QEuler(builder, loc, qubit, matrix, 0, true, basis);
    emitGPhaseIfNeeded(builder, loc, result->globalPhase);
    return result->qubit;
  }
  Value rotationAngle = std::get<Value>(angle);
  if (basis.singleQubit == SingleQubitBasis::U) {
    auto zero = mqt::constantFromScalar(builder, loc, 0.);
    auto halfPi = mqt::constantFromScalar(builder, loc, std::numbers::pi / 2.);
    auto negativeHalfPi =
        mqt::constantFromScalar(builder, loc, -std::numbers::pi / 2.);
    if (axis == PauliAxis::X) {
      return UOp::create(builder, loc, qubit, rotationAngle, negativeHalfPi,
                         halfPi)
          .getQubitOut();
    }
    if (axis == PauliAxis::Y) {
      return UOp::create(builder, loc, qubit, rotationAngle, zero, zero)
          .getQubitOut();
    }
    auto half = mqt::constantFromScalar(builder, loc, -0.5);
    emitGPhaseIfNeeded(
        builder, loc,
        builder.createOrFold<arith::MulFOp>(loc, rotationAngle, half));
    return UOp::create(builder, loc, qubit, zero, zero, rotationAngle)
        .getQubitOut();
  }
  if (basis.singleQubit == SingleQubitBasis::R && axis != PauliAxis::Z) {
    return ROp::create(builder, loc, qubit, rotationAngle,
                       mqt::constantFromScalar(
                           builder, loc,
                           axis == PauliAxis::X ? 0. : std::numbers::pi / 2.))
        .getQubitOut();
  }
  const bool supportsX = basis.singleQubit == SingleQubitBasis::ZXZ ||
                         basis.singleQubit == SingleQubitBasis::XZX ||
                         basis.singleQubit == SingleQubitBasis::XYX;
  const bool supportsY = basis.singleQubit == SingleQubitBasis::ZYZ ||
                         basis.singleQubit == SingleQubitBasis::XYX;
  const bool supportsZ = basis.singleQubit == SingleQubitBasis::ZYZ ||
                         basis.singleQubit == SingleQubitBasis::ZXZ ||
                         basis.singleQubit == SingleQubitBasis::XZX ||
                         basis.singleQubit == SingleQubitBasis::ZSXX;
  if (axis == PauliAxis::X && supportsX) {
    return RXOp::create(builder, loc, qubit, rotationAngle).getQubitOut();
  }
  if (axis == PauliAxis::Y && supportsY) {
    return RYOp::create(builder, loc, qubit, rotationAngle).getQubitOut();
  }
  if (axis == PauliAxis::Z && supportsZ) {
    return RZOp::create(builder, loc, qubit, rotationAngle).getQubitOut();
  }
  // Pick a native rotation and a constant frame; rotationAngle remains
  // untouched.
  const auto nativeAxis = supportsY || basis.singleQubit == SingleQubitBasis::R
                              ? PauliAxis::Y
                          : supportsX ? PauliAxis::X
                                      : PauliAxis::Z;
  const auto frame = pauliFrame(axis, nativeAxis);
  const auto before = synthesizeUnitary1QEuler(builder, loc, qubit,
                                               frame.adjoint(), 0, true, basis);
  emitGPhaseIfNeeded(builder, loc, before->globalPhase);
  qubit = synthesizePauliRotation1Q(builder, loc, before->qubit, nativeAxis,
                                    rotationAngle, basis);
  const auto after =
      synthesizeUnitary1QEuler(builder, loc, qubit, frame, 0, true, basis);
  emitGPhaseIfNeeded(builder, loc, after->globalPhase);
  return after->qubit;
}

/// Known ZYZ parameters need only affine arithmetic after angle normalization.
static std::array<RotationParameter, 4>
directEulerAngles(OpBuilder& builder, Location loc,
                  UnitaryOpInterface operation, SingleQubitBasis basis) {
  const auto parameter = [&](unsigned index) -> RotationParameter {
    auto angle = operation.getParameter(index);
    if (index == 0 && basis != SingleQubitBasis::ZSXX &&
        isa<RXOp, RYOp, ROp, UOp>(operation.getOperation()) &&
        !mqt::valueToConstantDouble(angle)) {
      return angle;
    }
    return normalizeRotationParameter(builder, loc, angle);
  };
  const auto add = [&](const RotationParameter& a, const RotationParameter& b) {
    return addRotationParameters(builder, loc, a, b);
  };
  const auto scale = [&](const RotationParameter& value,
                         double factor) -> RotationParameter {
    return mqt::scaleParameter(builder, loc, value, factor);
  };
  constexpr double halfPi = std::numbers::pi / 2.;
  std::array<RotationParameter, 4> result{0., 0., 0., 0.};
  auto& [theta, phi, lambda, phase] = result;
  Operation* op = operation.getOperation();
  if (isa<RXOp>(op)) {
    theta = parameter(0);
    phi = -halfPi;
    lambda = halfPi;
  } else if (isa<RYOp>(op)) {
    theta = parameter(0);
  } else if (isa<RZOp, POp>(op)) {
    lambda = parameter(0);
    if (isa<POp>(op) && basis != SingleQubitBasis::U) {
      phase = scale(lambda, 0.5);
    }
  } else if (isa<ROp>(op)) {
    theta = parameter(0);
    phi = add(parameter(1), -halfPi);
    lambda = scale(phi, -1.);
  } else {
    const bool u2 = isa<U2Op>(op);
    theta = u2 ? RotationParameter{halfPi} : parameter(0);
    phi = parameter(u2 ? 0 : 1);
    lambda = parameter(u2 ? 1 : 2);
    if (basis != SingleQubitBasis::U) {
      phase = scale(add(phi, lambda), 0.5);
    }
  }
  if (basis == SingleQubitBasis::ZXZ) {
    phi = add(phi, halfPi);
    lambda = add(lambda, -halfPi);
  } else if (basis == SingleQubitBasis::U) {
    // P/U2 already have U's phase; R/RX/RY have cancelling outer angles.
    phase = isa<RZOp>(op) ? scale(lambda, -0.5) : RotationParameter{0.};
  }
  return result;
}

std::optional<std::array<RotationParameter, 4>>
zyzAnglesFromOperation(OpBuilder& builder, Location loc,
                       UnitaryOpInterface operation) {
  if (canSynthesizeParameterizedUnitary1Q(operation.getOperation())) {
    return directEulerAngles(builder, loc, operation, SingleQubitBasis::ZYZ);
  }
  if (const auto matrix = operation.getUnitaryMatrix<Matrix2x2>()) {
    const auto angles = anglesFromUnitary(*matrix, SingleQubitBasis::ZYZ);
    return std::array<RotationParameter, 4>{
        angles.theta,
        angles.phi,
        angles.lambda,
        angles.phase,
    };
  }
  return std::nullopt;
}

bool canSynthesizeParameterizedUnitary1Q(Operation* op) {
  return op != nullptr && isa<RXOp, RYOp, RZOp, POp, ROp, U2Op, UOp>(op);
}

void synthesizeParameterizedUnitary1Q(
    RewriterBase& rewriter, Operation* op,
    const CompilerTarget::SynthesisBasis& basis) {
  assert(canSynthesizeParameterizedUnitary1Q(op));
  if (isSingleQubitBasisGate(op, basis.singleQubit)) {
    return;
  }
  auto unitary = cast<UnitaryOpInterface>(op);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  auto loc = op->getLoc();
  Value qubit = unitary.getInputQubit(0);
  const auto rotate = [&](PauliAxis axis, RotationParameter angle) {
    qubit = synthesizePauliRotation1Q(rewriter, loc, qubit, axis, angle, basis);
  };
  const auto phaseHalf = [&](Value angle) {
    auto half = mqt::constantFromScalar(rewriter, loc, 0.5);
    emitGPhaseIfNeeded(rewriter, loc,
                       rewriter.createOrFold<arith::MulFOp>(loc, angle, half));
  };
  if (basis.singleQubit == SingleQubitBasis::U ||
      basis.singleQubit == SingleQubitBasis::ZYZ ||
      basis.singleQubit == SingleQubitBasis::ZXZ ||
      basis.singleQubit == SingleQubitBasis::R ||
      basis.singleQubit == SingleQubitBasis::ZSXX) {
    qubit = emitParameterizedEulerAngles(
        rewriter, loc, qubit,
        directEulerAngles(rewriter, loc, unitary, basis.singleQubit), basis);
  } else if (const auto rotations = getPauliRotations(op)) {
    auto outputs = emitPauliRotations(rewriter, op, *rotations, basis, false);
    qubit = outputs.front();
  } else if (auto rotation = dyn_cast<ROp>(op)) {
    auto negative =
        rewriter.createOrFold<arith::NegFOp>(loc, rotation.getPhi());
    rotate(PauliAxis::Z, negative);
    rotate(PauliAxis::X, rotation.getTheta());
    rotate(PauliAxis::Z, rotation.getPhi());
  } else {
    const bool u2 = isa<U2Op>(op);
    auto phi = unitary.getParameter(u2 ? 0 : 1);
    auto lambda = unitary.getParameter(u2 ? 1 : 2);
    rotate(PauliAxis::Z, lambda);
    rotate(PauliAxis::Y, u2 ? RotationParameter{std::numbers::pi / 2.}
                            : RotationParameter{unitary.getParameter(0)});
    rotate(PauliAxis::Z, phi);
    phaseHalf(phi);
    phaseHalf(lambda);
  }
  rewriter.replaceOp(op, qubit);
}

} // namespace mlir::qco::decomposition
