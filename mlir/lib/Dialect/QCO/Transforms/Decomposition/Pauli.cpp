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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <numbers>
#include <optional>
#include <tuple>
#include <utility>
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

namespace {

/// E^dagger maps one local rotation on each wire to these commuting products.
struct CliffordPaulis {
  std::array<PauliAxis, 2> first;
  std::array<PauliAxis, 2> second;
  std::array<PauliAxis, 2> local;
  double firstSign = 1.;
};

} // namespace

static CliffordPaulis conjugatedPaulis(CompilerTarget::GateKind gate) {
  using enum PauliAxis;
  using Gate = CompilerTarget::GateKind;
  switch (gate) {
  case Gate::CX:
    return {.first = {X, X}, .second = {Z, Z}, .local = {X, Z}};
  case Gate::CZ:
    return {.first = {X, Z}, .second = {Z, X}, .local = {X, X}};
  case Gate::ECR:
    return {
        .first = {X, X},
        .second = {Z, Y},
        .local = {Y, Z},
        .firstSign = -1.,
    };
  case Gate::RZX:
    return {.first = {X, X}, .second = {Z, Y}, .local = {Y, Z}};
  case Gate::ISWAP:
    return {.first = {Z, X}, .second = {X, Z}, .local = {Y, Y}};
  case Gate::RXX:
    return {.first = {Y, X}, .second = {X, Y}, .local = {Z, Z}};
  case Gate::RYY:
    return {.first = {Z, Y}, .second = {Y, Z}, .local = {X, X}};
  case Gate::RZZ:
    return {.first = {X, Z}, .second = {Z, X}, .local = {Y, Y}};
  default:
    llvm_unreachable("unsupported fixed synthesis entangler");
  }
}

/// Emit C E^dagger (R_A(a) tensor R_B(b)) E C^dagger. Either angle
/// may be absent. The same sandwich handles one or two commuting generators.
static SmallVector<Value, 2>
emitCliffordSandwich(RewriterBase& rewriter, Location loc, Value wire0,
                     Value wire1, std::array<Matrix2x2, 2> frames,
                     std::array<Value, 2> angles,
                     const CompilerTarget::SynthesisBasis& basis) {
  auto fixedBasis = basis;
  fixedBasis.entangler->angles = CompilerTarget::AngleSupport::Fixed;
  const auto identity = Matrix2x2::identity();
  const auto axes = conjugatedPaulis(basis.entangler->gate);
  const TwoQubitNativeDecomposition before{
      .numBasisUses = 1,
      .singleQubitFactors =
          {
              frames[1].adjoint(),
              frames[0].adjoint(),
              identity,
              identity,
          },
  };
  double phase = 0.;
  /// E^dagger = D E. D is identity for CX/CZ/ECR, ZZ for iSWAP,
  /// and i times the generator for a fixed Pauli rotation at pi/2.
  if (basis.entangler->gate == CompilerTarget::GateKind::ISWAP) {
    frames[0] = frames[0] * ZOp::getUnitaryMatrix();
    frames[1] = frames[1] * ZOp::getUnitaryMatrix();
  } else if (basis.entangler->gate == CompilerTarget::GateKind::RXX ||
             basis.entangler->gate == CompilerTarget::GateKind::RYY ||
             basis.entangler->gate == CompilerTarget::GateKind::RZX ||
             basis.entangler->gate == CompilerTarget::GateKind::RZZ) {
    const auto nativeAxes = pauliAxes(basis.entangler->gate);
    frames[0] = frames[0] * pauliMatrix(nativeAxes[0]);
    frames[1] = frames[1] * pauliMatrix(nativeAxes[1]);
    phase += std::numbers::pi / 2.;
  }
  const TwoQubitNativeDecomposition after{
      .numBasisUses = 1,
      .singleQubitFactors = {identity, identity, frames[1], frames[0]},
  };
  const auto first =
      emitUnitary2QWeyl(rewriter, loc, wire0, wire1, before, fixedBasis);
  std::array wires{first.qubit0, first.qubit1};
  if (angles[0] && axes.firstSign < 0.) {
    angles[0] = rewriter.createOrFold<arith::NegFOp>(loc, angles[0]);
  }
  for (size_t i = 0; i < wires.size(); ++i) {
    if (angles[i]) {
      wires[i] = synthesizePauliRotation1Q(rewriter, loc, wires[i],
                                           axes.local[i], angles[i], basis);
    }
  }
  const auto second =
      emitUnitary2QWeyl(rewriter, loc, wires[0], wires[1], after, fixedBasis);
  emitGPhaseIfNeeded(rewriter, loc,
                     phase + first.globalPhase + second.globalPhase);
  return {second.qubit0, second.qubit1};
}

static SmallVector<Value, 2>
emitPauliSequence(RewriterBase& rewriter, Location loc, ValueRange inputs,
                  const PauliRotationSequence& sequence,
                  const CompilerTarget::SynthesisBasis& basis, bool reverse) {
  auto wires = llvm::to_vector<2>(inputs);
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
      const mqt::FloatExpression normalized(
          rewriter, loc, normalizeRotationParameter(rewriter, loc, angle));
      const auto scalar = [&](double value) {
        return mqt::FloatExpression::constant(rewriter, loc, value);
      };
      const auto reduced = normalized.tan().atan();
      const auto half = scalar(0.5);
      const auto ratio = (reduced * half).sin() / reduced.cos().pow(half);
      const auto alpha = ratio.atan();
      const auto beta =
          scalar(-2.) * (ratio * scalar(std::numbers::sqrt2)).atan();
      const auto turns = normalized - reduced;
      const auto frame0 = pauliFrame(axis0, PauliAxis::Y);
      const auto frame1 = pauliFrame(axis1, PauliAxis::Z);
      emitFactor(wire0, frame0.adjoint());
      emitFactor(wire1, frame1.adjoint());
      const auto rotate = [&](Value& wire, PauliAxis axis,
                              mqt::FloatExpression value) {
        wire = synthesizePauliRotation1Q(rewriter, loc, wire, axis,
                                         value.getValue(), basis);
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
      GPhaseOp::create(rewriter, loc, (turns * half).getValue());
      continue;
    }
    /// Bounded rotations supply the fixed Clifford endpoint at pi/2.
    const auto axes = conjugatedPaulis(basis.entangler->gate);
    const auto outputs = emitCliffordSandwich(
        rewriter, loc, wire0, wire1,
        {pauliFrame(axis0, axes.second[0]), pauliFrame(axis1, axes.second[1])},
        {Value{}, angle}, basis);
    wire0 = outputs[0];
    wire1 = outputs[1];
  }
  if (sequence.globalPhaseScale != 0.) {
    emitGPhaseIfNeeded(rewriter, loc, scaledAngle(sequence.globalPhaseScale));
  }
  emitGPhaseIfNeeded(rewriter, loc, globalPhase);
  return wires;
}

static void hoistPauliAngle(RewriterBase& rewriter, Operation* operation) {
  if (auto controlled = dyn_cast<CtrlOp>(operation)) {
    auto body =
        mqt::getSoleBodyUnitary<UnitaryOpInterface>(*controlled.getBody());
    mqt::hoistSupportingOpsBefore(*controlled.getBody(), body.getOperation(),
                                  controlled, rewriter);
  }
}

void mergeDiagonalRotations(RewriterBase& rewriter, ModuleOp moduleOp) {
  moduleOp->walk<WalkOrder::PreOrder>([&](Operation* parent) {
    if (isa<UnitaryOpInterface>(parent)) {
      return WalkResult::skip();
    }
    for (Region& region : parent->getRegions()) {
      for (Block& block : region) {
        struct Group {
          SmallVector<Operation*> operations;
          SmallVector<RotationAngleTerm> angles;
        };
        DenseMap<Value, size_t> wires;
        DenseMap<std::tuple<size_t, size_t, unsigned>, size_t> indices;
        SmallVector<Group, 4> groups;
        size_t nextWire = 0;
        const auto flush = [&] {
          for (auto& group : groups) {
            if (group.operations.size() < 2) {
              continue;
            }
            for (auto* operation : group.operations) {
              hoistPauliAngle(rewriter, operation);
            }
            simplifyRotationAngles(group.angles);
            auto* last = group.operations.back();
            if (!group.angles.empty()) {
              rewriter.setInsertionPoint(last);
              Value angle =
                  emitRotationAngleSum(rewriter, last->getLoc(), group.angles);
              auto gate = cast<UnitaryOpInterface>(last);
              if (auto controlled = dyn_cast<CtrlOp>(last)) {
                gate = mqt::getSoleBodyUnitary<UnitaryOpInterface>(
                    *controlled.getBody());
              }
              rewriter.modifyOpInPlace(gate, [&] {
                TypeSwitch<Operation*>(gate).Case<RZZOp, RZOp, POp>(
                    [&](auto rotation) {
                      rotation.getThetaMutable().assign(angle);
                    });
              });
              group.operations.pop_back();
            }
            for (auto* operation : llvm::reverse(group.operations)) {
              rewriter.replaceOp(
                  operation,
                  cast<UnitaryOpInterface>(operation).getInputQubits());
            }
          }
          groups.clear();
          indices = decltype(indices){};
          wires = decltype(wires){};
        };
        for (Operation& operation : llvm::make_early_inc_range(block)) {
          auto gate = dyn_cast<UnitaryOpInterface>(operation);
          if (!gate) {
            if (!isMemoryEffectFree(&operation) ||
                operation.getNumRegions() != 0) {
              flush();
            }
            continue;
          }
          auto body = gate;
          if (auto controlled = dyn_cast<CtrlOp>(operation)) {
            body = controlled.getNumControls() == 1 &&
                           controlled.getNumTargets() == 1
                       ? mqt::getSoleBodyUnitary<UnitaryOpInterface>(
                             *controlled.getBody())
                       : UnitaryOpInterface{};
          }
          if (!body || !isa<IdOp, ZOp, SOp, SdgOp, TOp, TdgOp, RZOp, POp, RZZOp,
                            GPhaseOp>(body.getOperation())) {
            flush();
            continue;
          }
          SmallVector<size_t, 2> operands;
          for (Value input : gate.getInputQubits()) {
            auto [position, inserted] = wires.try_emplace(input, nextWire);
            if (inserted) {
              ++nextWire;
            }
            operands.push_back(position->second);
            wires.erase(position);
          }
          for (auto [output, wire] :
               llvm::zip_equal(gate.getOutputQubits(), operands)) {
            wires.try_emplace(output, wire);
          }
          if (operands.size() != 2 ||
              !isa<RZZOp, RZOp, POp>(body.getOperation())) {
            continue;
          }
          /// Controlled RZ is directional; controlled P and RZZ are symmetric.
          const unsigned kind = isa<RZZOp>(body.getOperation())  ? 0
                                : isa<RZOp>(body.getOperation()) ? 1
                                                                 : 2;
          if (kind != 1 && operands[0] > operands[1]) {
            std::swap(operands[0], operands[1]);
          }
          auto [found, inserted] = indices.try_emplace(
              std::tuple{operands[0], operands[1], kind}, groups.size());
          if (inserted) {
            groups.emplace_back();
          }
          auto& group = groups[found->second];
          group.operations.push_back(&operation);
          group.angles.push_back(
              {.value = getPauliRotations(&operation)->angle});
        }
        flush();
      }
    }
    return WalkResult::advance();
  });
}

SmallVector<Value, 2>
emitPauliRotations(RewriterBase& rewriter, Operation* operation,
                   const PauliRotationSequence& sequence,
                   const CompilerTarget::SynthesisBasis& basis, bool reverse) {
  hoistPauliAngle(rewriter, operation);
  rewriter.setInsertionPoint(operation);
  return emitPauliSequence(rewriter, operation->getLoc(),
                           cast<UnitaryOpInterface>(operation).getInputQubits(),
                           sequence, basis, reverse);
}

std::optional<size_t>
pauliRotationEntanglerCount(const PauliRotationSequence& sequence,
                            CompilerTarget::Entangler entangler) {
  if (const auto angle = mqt::valueToConstantDouble(sequence.angle)) {
    for (const auto& term : sequence.rotations) {
      if (term.axes[0] != PauliAxis::I && term.axes[1] != PauliAxis::I) {
        const double value = *angle * term.angleScale;
        /// Let the Weyl planner shorten near-Clifford rotations using its
        /// fidelity policy, including negligible entangling angles.
        const double delta = std::remainder(value, std::numbers::pi / 2.);
        if (traceToFidelity(4. * std::cos(delta / 2.)) >=
            WEYL_DEFAULT_FIDELITY) {
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

namespace {

struct PauliGroup {
  std::array<PauliAxis, 2> axes;
  SmallVector<RotationAngleTerm> angle;

  [[nodiscard]] bool entangling() const {
    return axes[0] != PauliAxis::I && axes[1] != PauliAxis::I;
  }
};

} // namespace

static bool commute(std::array<PauliAxis, 2> first,
                    std::array<PauliAxis, 2> second) {
  const auto anticommute = [](PauliAxis a, PauliAxis b) {
    return a != PauliAxis::I && b != PauliAxis::I && a != b;
  };
  return anticommute(first[0], second[0]) == anticommute(first[1], second[1]);
}

/// Match two Pauli axes with a single Clifford frame; no Euler extraction.
static Matrix2x2 pauliPairFrame(PauliAxis first, PauliAxis second,
                                PauliAxis fromFirst, PauliAxis fromSecond) {
  auto frame = pauliFrame(first, fromFirst);
  const auto axis = pauliFrame(first);
  const auto quarter =
      axis * RZOp::unitaryMatrix(std::numbers::pi / 2.) * axis.adjoint();
  const auto wanted = pauliMatrix(second);
  for (size_t turn = 0; turn < 4; ++turn) {
    if ((frame * pauliMatrix(fromSecond) * frame.adjoint()).isApprox(wanted)) {
      return frame;
    }
    frame = quarter * frame;
  }
  llvm_unreachable("distinct Pauli axes admit a Clifford frame");
}

LogicalResult
fusePauliRotationRun(PatternRewriter& rewriter, Operation* head,
                     const CompilerTarget::SynthesisBasis& basis, bool reverse,
                     const CompilerTarget& target,
                     std::optional<ArrayRef<CompilerTarget::SiteId>> sites) {
  auto first = dyn_cast<UnitaryOpInterface>(head);
  if (!first || !first.isTwoQubit() || !basis.entangler) {
    return failure();
  }
  SmallVector<Operation*> operations;
  SmallVector<PauliGroup, 4> groups;
  SmallVector<RotationAngleTerm> phase;
  std::array wires{first.getInputQubit(0), first.getInputQubit(1)};
  size_t separateCost = 0;
  for (auto current = first;
       current && current->getBlock() == head->getBlock();) {
    auto sequence = getPauliRotations(current);
    if (!sequence) {
      break;
    }
    /// Numerical fusion owns constant heads; runtime runs may absorb constants.
    if (operations.empty() && mqt::valueToConstantDouble(sequence->angle)) {
      return failure();
    }
    const bool reversed = current.getInputQubit(0) != wires[0];
    if (reversed) {
      for (auto& term : sequence->rotations) {
        std::swap(term.axes[0], term.axes[1]);
      }
    }
    auto operationSites = sites ? llvm::to_vector<2>(*sites)
                                : SmallVector<CompilerTarget::SiteId, 2>{};
    if (reversed && sites) {
      std::swap(operationSites[0], operationSites[1]);
    }
    const auto cost =
        (sites ? target.supports(current, operationSites)
               : target.supports(current))
            ? std::optional<size_t>{1}
            : pauliRotationEntanglerCount(*sequence, *basis.entangler);
    if (!cost) {
      break;
    }
    if (llvm::any_of(sequence->rotations, [&](const auto& term) {
          return llvm::any_of(groups, [&](const auto& group) {
            return !commute(term.axes, group.axes);
          });
        })) {
      break;
    }
    for (const auto& term : sequence->rotations) {
      auto* found = llvm::find_if(
          groups, [&](const auto& group) { return group.axes == term.axes; });
      if (found == groups.end()) {
        groups.push_back({.axes = term.axes});
        found = std::prev(groups.end());
      }
      found->angle.push_back(
          {.value = sequence->angle, .scale = term.angleScale});
    }
    if (sequence->globalPhaseScale != 0.) {
      phase.push_back(
          {.value = sequence->angle, .scale = sequence->globalPhaseScale});
    }
    operations.push_back(current);
    separateCost += *cost;
    wires = {
        current.getOutputForInput(wires[0]),
        current.getOutputForInput(wires[1]),
    };
    auto next = dyn_cast<UnitaryOpInterface>(*wires[0].user_begin());
    current = next && next.isTwoQubit() &&
                      next.getOperation() == *wires[1].user_begin()
                  ? next
                  : UnitaryOpInterface{};
  }
  if (operations.size() < 2) {
    return failure();
  }
  SmallVector<const PauliGroup*, 3> entangling;
  size_t combinedCost = 0;
  for (auto& group : groups) {
    simplifyRotationAngles(group.angle);
    if (group.entangling() && !group.angle.empty()) {
      entangling.push_back(&group);
      combinedCost +=
          basis.entangler->parameterized() &&
                  (basis.entangler->angles ==
                       CompilerTarget::AngleSupport::Unrestricted ||
                   llvm::all_of(group.angle,
                                [](const auto& term) {
                                  return mqt::valueToConstantDouble(term.value)
                                      .has_value();
                                }))
              ? 1
              : 2;
    }
  }
  const bool shared =
      entangling.size() >= 2 && combinedCost > entangling.size() &&
      basis.entangler->gate != CompilerTarget::GateKind::SQRTISWAP;
  if ((shared ? entangling.size() : combinedCost) >= separateCost) {
    return failure();
  }
  for (auto* operation : operations) {
    hoistPauliAngle(rewriter, operation);
  }
  rewriter.setInsertionPoint(operations.back());
  auto loc = head->getLoc();
  auto outputs = llvm::to_vector<2>(first.getInputQubits());
  if (shared && entangling.size() == 3) {
    const size_t index0 = reverse ? 1 : 0;
    const size_t index1 = reverse ? 0 : 1;
    const std::array frames{
        pauliPairFrame(entangling[0]->axes[index0], entangling[1]->axes[index0],
                       PauliAxis::X, PauliAxis::Y),
        pauliPairFrame(entangling[0]->axes[index1], entangling[1]->axes[index1],
                       PauliAxis::X, PauliAxis::Y),
    };
    std::array angles{
        emitRotationAngleSum(rewriter, loc, entangling[0]->angle),
        emitRotationAngleSum(rewriter, loc, entangling[1]->angle),
        emitRotationAngleSum(rewriter, loc, entangling[2]->angle),
    };
    const auto z = ZOp::getUnitaryMatrix();
    const bool firstPositive =
        (frames[0] * z * frames[0].adjoint())
            .isApprox(pauliMatrix(entangling[2]->axes[index0]));
    const bool secondPositive =
        (frames[1] * z * frames[1].adjoint())
            .isApprox(pauliMatrix(entangling[2]->axes[index1]));
    if (firstPositive != secondPositive) {
      angles[2] = rewriter.createOrFold<arith::NegFOp>(loc, angles[2]);
    }
    const auto result = cachedNativeBasisDecomposer(basis.entangler->gate)
                            .emitCartan(rewriter, loc, outputs[index0],
                                        outputs[index1], angles, frames, basis);
    outputs[index0] = result[0];
    outputs[index1] = result[1];
  } else if (shared) {
    const auto axes = conjugatedPaulis(basis.entangler->gate);
    const size_t index0 = reverse ? 1 : 0;
    const size_t index1 = reverse ? 0 : 1;
    const auto result = emitCliffordSandwich(
        rewriter, loc, outputs[index0], outputs[index1],
        {
            pauliPairFrame(entangling[0]->axes[index0],
                           entangling[1]->axes[index0], axes.first[0],
                           axes.second[0]),
            pauliPairFrame(entangling[0]->axes[index1],
                           entangling[1]->axes[index1], axes.first[1],
                           axes.second[1]),
        },
        {
            emitRotationAngleSum(rewriter, loc, entangling[0]->angle),
            emitRotationAngleSum(rewriter, loc, entangling[1]->angle),
        },
        basis);
    outputs[index0] = result[0];
    outputs[index1] = result[1];
  }
  for (const auto& group : groups) {
    if (!group.angle.empty() && !(shared && group.entangling())) {
      outputs = emitPauliSequence(
          rewriter, loc, outputs,
          {
              .angle = emitRotationAngleSum(rewriter, loc, group.angle),
              .rotations = {{.axes = group.axes}},
          },
          basis, reverse);
    }
  }
  simplifyRotationAngles(phase);
  if (!phase.empty()) {
    emitGPhaseIfNeeded(rewriter, loc,
                       emitRotationAngleSum(rewriter, loc, phase));
  }
  rewriter.replaceAllUsesWith(wires[0], outputs[0]);
  rewriter.replaceAllUsesWith(wires[1], outputs[1]);
  for (auto* operation : llvm::reverse(operations)) {
    rewriter.eraseOp(operation);
  }
  return success();
}

} // namespace mlir::qco::decomposition
