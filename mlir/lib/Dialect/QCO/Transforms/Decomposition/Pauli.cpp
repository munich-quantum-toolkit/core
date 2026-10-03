/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Transforms/Decomposition/Pauli.h"

#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/MQT/Utils/Modifiers.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mqt/Dialect/QCO/Transforms/Decomposition/Weyl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <optional>
#include <variant>

namespace mlir::qco::decomposition {

Matrix2x2 pauliFrame(PauliAxis axis, PauliAxis from) {
  if (axis == from || axis == PauliAxis::I) {
    return Matrix2x2::identity();
  }
  if (from != PauliAxis::Z) {
    if (axis == PauliAxis::Z) {
      return pauliFrame(from).adjoint();
    }
    return RZOp::unitaryMatrix(axis == PauliAxis::Y ? std::numbers::pi / 2.
                                                    : -std::numbers::pi / 2.);
  }
  switch (axis) {
  case PauliAxis::X:
    return RYOp::unitaryMatrix(std::numbers::pi / 2.);
  case PauliAxis::Y:
    return RXOp::unitaryMatrix(-std::numbers::pi / 2.);
  case PauliAxis::I:
  case PauliAxis::Z:
    return Matrix2x2::identity();
  }
  llvm_unreachable("unknown Pauli axis");
}

FoldedPauliAngle foldPauliAngle(double angle) {
  /// Reduce with trigonometry before subtracting pi, including large angles.
  const double principal = std::abs(angle) <= 2. * std::numbers::pi
                               ? angle
                               : 4. * std::atan(std::tan(angle / 4.));
  const double folded = std::remainder(principal, std::numbers::pi);
  const auto turns =
      static_cast<int>(std::round((principal - folded) / std::numbers::pi));
  return {
      .angle = std::abs(folded),
      .phase = -turns * std::numbers::pi / 2.,
      .product = turns % 2 != 0,
      .negate = folded < 0.,
  };
}

static Matrix2x2 pauliMatrix(PauliAxis axis) {
  const auto frame = pauliFrame(axis);
  return frame * ZOp::getUnitaryMatrix() * frame.adjoint();
}

std::array<PauliAxis, 2> pauliAxes(CompilerTarget::GateKind gate) {
  using enum PauliAxis;
  switch (gate) {
  case CompilerTarget::GateKind::RX:
    return {X, I};
  case CompilerTarget::GateKind::RY:
    return {Y, I};
  case CompilerTarget::GateKind::RZ:
    return {Z, I};
  case CompilerTarget::GateKind::RXX:
    return {X, X};
  case CompilerTarget::GateKind::RYY:
    return {Y, Y};
  case CompilerTarget::GateKind::RZX:
    return {Z, X};
  case CompilerTarget::GateKind::RZZ:
    return {Z, Z};
  default:
    llvm_unreachable("gate is not a Pauli rotation");
  }
}

Operation* emitPauliRotation2Q(OpBuilder& builder, Location loc, Value qubit0,
                               Value qubit1, CompilerTarget::GateKind gate,
                               std::variant<double, Value> angle) {
  Value theta = mqt::variantToValue(builder, loc, angle);
  switch (gate) {
  case CompilerTarget::GateKind::RXX:
    return RXXOp::create(builder, loc, qubit0, qubit1, theta);
  case CompilerTarget::GateKind::RYY:
    return RYYOp::create(builder, loc, qubit0, qubit1, theta);
  case CompilerTarget::GateKind::RZX:
    return RZXOp::create(builder, loc, qubit0, qubit1, theta);
  case CompilerTarget::GateKind::RZZ:
    return RZZOp::create(builder, loc, qubit0, qubit1, theta);
  default:
    llvm_unreachable("gate is not a two-qubit Pauli rotation");
  }
}

std::optional<PauliRotationSequence> getPauliRotations(Operation* operation) {
  auto unitary = dyn_cast<UnitaryOpInterface>(operation);
  if (!unitary) {
    return std::nullopt;
  }
  auto controlled = dyn_cast<CtrlOp>(operation);
  if (controlled) {
    if (controlled.getNumControls() != 1 || controlled.getNumTargets() != 1) {
      return std::nullopt;
    }
    unitary =
        mqt::getSoleBodyUnitary<UnitaryOpInterface>(*controlled.getBody());
    if (!unitary || !unitary.isSingleQubit()) {
      return std::nullopt;
    }
  }
  const auto gate =
      TypeSwitch<Operation*, std::optional<CompilerTarget::GateKind>>(
          unitary.getOperation())
          .Case([](RXOp) { return CompilerTarget::GateKind::RX; })
          .Case([](RYOp) { return CompilerTarget::GateKind::RY; })
          .Case<RZOp, POp>([](auto) { return CompilerTarget::GateKind::RZ; })
          .Case([](RXXOp) { return CompilerTarget::GateKind::RXX; })
          .Case([](RYYOp) { return CompilerTarget::GateKind::RYY; })
          .Case([](RZXOp) { return CompilerTarget::GateKind::RZX; })
          .Case([](RZZOp) { return CompilerTarget::GateKind::RZZ; })
          .Default([](auto) { return std::nullopt; });
  if (!gate) {
    return std::nullopt;
  }
  const auto axes = pauliAxes(*gate);
  const bool phase = isa<POp>(unitary.getOperation());
  PauliRotationSequence result{
      .angle = unitary.getParameter(0),
      .rotations = {{.axes = axes}},
      .globalPhaseScale = phase ? 0.5 : 0.,
  };
  if (controlled) {
    using enum PauliAxis;
    result.rotations = {
        {.axes = {I, axes[0]}, .angleScale = 0.5},
        {.axes = {Z, axes[0]}, .angleScale = -0.5},
    };
    if (phase) {
      result.rotations.push_back({.axes = {Z, I}, .angleScale = 0.5});
      result.globalPhaseScale = 0.25;
    }
  }
  return result;
}

/// For each fixed Clifford E, E^dagger (I tensor Q) E = A tensor B.
/// The first two axes are A, B; the third is the local rotation axis Q.
static std::array<PauliAxis, 3>
conjugatedPauliAxes(CompilerTarget::GateKind gate) {
  using enum PauliAxis;
  using Gate = CompilerTarget::GateKind;
  switch (gate) {
  case Gate::CX:
    return {Z, Z, Z};
  case Gate::CZ:
    return {Z, X, X};
  case Gate::ECR:
  case Gate::RZX:
    return {Z, Y, Z};
  case Gate::ISWAP:
    return {X, Z, Y};
  case Gate::RXX:
    return {X, Y, Z};
  case Gate::RYY:
    return {Y, Z, X};
  case Gate::RZZ:
    return {Z, X, Y};
  default:
    llvm_unreachable("unsupported fixed synthesis entangler");
  }
}

SmallVector<Value, 2>
emitPauliRotations(RewriterBase& rewriter, Operation* operation,
                   const PauliRotationSequence& sequence,
                   const CompilerTarget::SynthesisBasis& basis, bool reverse) {
  auto unitary = cast<UnitaryOpInterface>(operation);
  if (auto controlled = dyn_cast<CtrlOp>(operation)) {
    auto body =
        mqt::getSoleBodyUnitary<UnitaryOpInterface>(*controlled.getBody());
    mqt::hoistSupportingOpsBefore(*controlled.getBody(), body.getOperation(),
                                  controlled, rewriter);
  }
  rewriter.setInsertionPoint(operation);
  auto loc = operation->getLoc();
  auto wires = llvm::to_vector<2>(unitary.getInputQubits());
  double globalPhase = 0.;
  const auto scaledAngle = [&](double scale) -> Value {
    if (scale == 1.) {
      return sequence.angle;
    }
    auto factor = mqt::constantFromScalar(rewriter, loc, scale);
    return arith::MulFOp::create(rewriter, loc, sequence.angle, factor);
  };
  const auto emitFactor = [&](Value& wire, const Matrix2x2& matrix) {
    const auto synthesized =
        synthesizeUnitary1QEuler(rewriter, loc, wire, matrix, 0, true, basis);
    wire = synthesized->qubit;
    globalPhase += synthesized->globalPhase;
  };
  for (const auto& term : sequence.rotations) {
    auto angle = scaledAngle(term.angleScale);
    if (term.axes[0] == PauliAxis::I || term.axes[1] == PauliAxis::I) {
      const size_t wire = term.axes[0] == PauliAxis::I ? 1 : 0;
      wires[wire] = synthesizePauliRotation1Q(rewriter, loc, wires[wire],
                                              term.axes[wire], angle, basis);
      continue;
    }
    assert(basis.entangler && "two-qubit synthesis requires an entangler");
    Value& wire0 = wires[reverse ? 1 : 0];
    Value& wire1 = wires[reverse ? 0 : 1];
    const auto axis0 = term.axes[reverse ? 1 : 0];
    const auto axis1 = term.axes[reverse ? 0 : 1];
    const bool direct = basis.entangler->parameterized() &&
                        (basis.entangler->angles ==
                             CompilerTarget::AngleSupport::Unrestricted ||
                         mqt::valueToConstantDouble(sequence.angle));
    if (direct) {
      const auto nativeAxes = pauliAxes(basis.entangler->gate);
      auto frame0 = pauliFrame(axis0, nativeAxes[0]);
      auto frame1 = pauliFrame(axis1, nativeAxes[1]);
      auto before0 = frame0.adjoint();
      auto before1 = frame1.adjoint();
      if (basis.entangler->angles ==
          CompilerTarget::AngleSupport::ZeroToHalfPi) {
        const auto constant = mqt::valueToConstantDouble(sequence.angle);
        assert(constant && "bounded synthesis requires a known angle");
        const auto folded = foldPauliAngle(*constant * term.angleScale);
        angle = mqt::constantFromScalar(rewriter, loc, folded.angle);
        globalPhase += folded.phase;
        if (folded.product) {
          frame0 = frame0 * pauliMatrix(nativeAxes[0]);
          frame1 = frame1 * pauliMatrix(nativeAxes[1]);
        }
        if (folded.negate) {
          const auto flip = pauliMatrix(
              nativeAxes[0] == PauliAxis::Z ? PauliAxis::X : PauliAxis::Z);
          before0 = flip * before0;
          frame0 = frame0 * flip;
        }
      }
      emitFactor(wire0, before0);
      emitFactor(wire1, before1);
      auto* native = emitPauliRotation2Q(rewriter, loc, wire0, wire1,
                                         basis.entangler->gate, angle);
      wire0 = native->getResult(0);
      wire1 = native->getResult(1);
      emitFactor(wire0, frame0);
      emitFactor(wire1, frame1);
      continue;
    }
    if (basis.entangler->gate == CompilerTarget::GateKind::SQRTISWAP) {
      /// S^dagger (I X) S = (I X - Y Z)/sqrt(2). Symmetric RX
      /// corrections cancel the local X component, leaving R_YZ(theta).
      /// Reduce modulo pi; the removed turns are local Pauli rotations.
      auto normalized = mqt::variantToValue(
          rewriter, loc, normalizeRotationParameter(rewriter, loc, angle));
      auto reduced = rewriter.createOrFold<math::AtanOp>(
          loc, rewriter.createOrFold<math::TanOp>(loc, normalized));
      auto half = mqt::constantFromScalar(rewriter, loc, 0.5);
      auto sine = rewriter.createOrFold<math::SinOp>(
          loc, rewriter.createOrFold<arith::MulFOp>(loc, reduced, half));
      auto cosine = rewriter.createOrFold<math::CosOp>(loc, reduced);
      auto root = rewriter.createOrFold<math::PowFOp>(loc, cosine, half);
      auto ratio = rewriter.createOrFold<arith::DivFOp>(loc, sine, root);
      auto alpha = rewriter.createOrFold<math::AtanOp>(loc, ratio);
      auto beta = rewriter.createOrFold<arith::MulFOp>(
          loc, mqt::constantFromScalar(rewriter, loc, -2.),
          rewriter.createOrFold<math::AtanOp>(
              loc, rewriter.createOrFold<arith::MulFOp>(
                       loc, ratio,
                       mqt::constantFromScalar(rewriter, loc,
                                               std::numbers::sqrt2))));
      auto turns =
          rewriter.createOrFold<arith::SubFOp>(loc, normalized, reduced);
      const auto frame0 = pauliFrame(axis0, PauliAxis::Y);
      const auto frame1 = pauliFrame(axis1, PauliAxis::Z);
      emitFactor(wire0, frame0.adjoint());
      emitFactor(wire1, frame1.adjoint());
      const auto rotate = [&](Value& wire, PauliAxis axis, Value value) {
        wire =
            synthesizePauliRotation1Q(rewriter, loc, wire, axis, value, basis);
      };
      const auto entangle = [&] {
        auto native = XXPlusYYOp::create(
            rewriter, loc, wire0, wire1,
            mqt::constantFromScalar(rewriter, loc, -std::numbers::pi / 2.),
            mqt::constantFromScalar(rewriter, loc, 0.));
        wire0 = native.getResult(0);
        wire1 = native.getResult(1);
      };
      rotate(wire1, PauliAxis::X, alpha);
      entangle();
      rotate(wire1, PauliAxis::X, beta);
      emitFactor(wire0, ZOp::getUnitaryMatrix());
      entangle();
      emitFactor(wire0, ZOp::getUnitaryMatrix());
      rotate(wire1, PauliAxis::X, alpha);
      rotate(wire0, PauliAxis::Y, turns);
      rotate(wire1, PauliAxis::Z, turns);
      emitFactor(wire0, frame0);
      emitFactor(wire1, frame1);
      GPhaseOp::create(rewriter, loc,
                       rewriter.createOrFold<arith::MulFOp>(loc, turns, half));
      continue;
    }
    /// Bounded native rotations include pi/2, which supplies the fixed
    /// Clifford primitive for angles that cannot be checked at compile time.
    auto fixedBasis = basis;
    fixedBasis.entangler->angles = CompilerTarget::AngleSupport::Fixed;
    const auto identity = Matrix2x2::identity();
    const auto axes = conjugatedPauliAxes(basis.entangler->gate);
    auto frame0 = pauliFrame(axis0, axes[0]);
    auto frame1 = pauliFrame(axis1, axes[1]);
    const TwoQubitNativeDecomposition before{
        .numBasisUses = 1,
        .singleQubitFactors =
            {
                frame1.adjoint(),
                frame0.adjoint(),
                identity,
                identity,
            },
    };
    /// E^dagger = D E. D is identity for CX/CZ/ECR, ZZ for iSWAP,
    /// and i times the generator for a fixed Pauli rotation at pi/2.
    if (basis.entangler->gate == CompilerTarget::GateKind::ISWAP) {
      frame0 = frame0 * ZOp::getUnitaryMatrix();
      frame1 = frame1 * ZOp::getUnitaryMatrix();
    } else if (basis.entangler->gate == CompilerTarget::GateKind::RXX ||
               basis.entangler->gate == CompilerTarget::GateKind::RYY ||
               basis.entangler->gate == CompilerTarget::GateKind::RZX ||
               basis.entangler->gate == CompilerTarget::GateKind::RZZ) {
      const auto nativeAxes = pauliAxes(basis.entangler->gate);
      frame0 = frame0 * pauliMatrix(nativeAxes[0]);
      frame1 = frame1 * pauliMatrix(nativeAxes[1]);
      globalPhase += std::numbers::pi / 2.;
    }
    const TwoQubitNativeDecomposition after{
        .numBasisUses = 1,
        .singleQubitFactors = {identity, identity, frame1, frame0},
    };
    const auto first =
        emitUnitary2QWeyl(rewriter, loc, wire0, wire1, before, fixedBasis);
    wire0 = first.qubit0;
    wire1 = synthesizePauliRotation1Q(rewriter, loc, first.qubit1, axes[2],
                                      angle, basis);
    const auto second =
        emitUnitary2QWeyl(rewriter, loc, wire0, wire1, after, fixedBasis);
    wire0 = second.qubit0;
    wire1 = second.qubit1;
    globalPhase += first.globalPhase + second.globalPhase;
  }
  if (sequence.globalPhaseScale != 0.) {
    GPhaseOp::create(rewriter, loc, scaledAngle(sequence.globalPhaseScale));
  }
  emitGPhaseIfNeeded(rewriter, loc, globalPhase);
  return wires;
}

std::optional<size_t>
pauliRotationEntanglerCount(const PauliRotationSequence& sequence,
                            CompilerTarget::Entangler entangler) {
  if (const auto angle = mqt::valueToConstantDouble(sequence.angle)) {
    for (const auto& term : sequence.rotations) {
      if (term.axes[0] != PauliAxis::I && term.axes[1] != PauliAxis::I) {
        const double value = *angle * term.angleScale;
        if (std::abs(std::sin(value)) <= MATRIX_TOLERANCE ||
            std::abs(std::cos(value)) <= MATRIX_TOLERANCE) {
          return std::nullopt;
        }
      }
    }
  }
  if (entangler.parameterized() &&
      (entangler.angles == CompilerTarget::AngleSupport::Unrestricted ||
       mqt::valueToConstantDouble(sequence.angle))) {
    return 1;
  }
  return 2;
}

} // namespace mlir::qco::decomposition
