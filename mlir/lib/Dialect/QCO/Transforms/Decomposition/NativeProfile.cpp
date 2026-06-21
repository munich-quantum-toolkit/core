/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/NativeProfile.h"

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

using mlir::qco::Matrix2x2;
using mlir::qco::Matrix4x4;
using mlir::qco::twoQubitControlledX01;
using mlir::qco::twoQubitControlledZ;

namespace mlir::qco::decomposition {

namespace {

[[nodiscard]] std::optional<NativeGateKind>
parseGateToken(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<NativeGateKind>>(name)
      .Case("u", NativeGateKind::U)
      .Case("x", NativeGateKind::X)
      .Case("sx", NativeGateKind::Sx)
      .Cases("rz", "p", NativeGateKind::Rz)
      .Case("rx", NativeGateKind::Rx)
      .Case("ry", NativeGateKind::Ry)
      .Case("r", NativeGateKind::R)
      .Case("cx", NativeGateKind::Cx)
      .Case("cz", NativeGateKind::Cz)
      .Case("rzz", NativeGateKind::Rzz)
      .Default(std::nullopt);
}

[[nodiscard]] std::optional<llvm::DenseSet<NativeGateKind>>
parseGateSet(llvm::StringRef nativeGates) {
  llvm::DenseSet<NativeGateKind> gates;
  llvm::SmallVector<llvm::StringRef> parts;
  nativeGates.split(parts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef part : parts) {
    const auto token = part.trim().lower();
    if (token.empty()) {
      continue;
    }
    const auto gate = parseGateToken(token);
    if (!gate) {
      return std::nullopt;
    }
    gates.insert(*gate);
  }
  return gates;
}

[[nodiscard]] SingleQubitEmitterSpec
makeEmitterSpec(SingleQubitMode mode, AxisPair axisPair = AxisPair::RxRz,
                bool supportsDirectRx = false) {
  return {
      .mode = mode, .axisPair = axisPair, .supportsDirectRx = supportsDirectRx};
}

void addEmitterIfAbsent(llvm::SmallVectorImpl<SingleQubitEmitterSpec>& emitters,
                        SingleQubitMode mode,
                        AxisPair axisPair = AxisPair::RxRz,
                        bool supportsDirectRx = false) {
  const bool present = llvm::any_of(emitters, [&](const auto& e) {
    return e.mode == mode && e.axisPair == axisPair &&
           e.supportsDirectRx == supportsDirectRx;
  });
  if (!present) {
    emitters.push_back(makeEmitterSpec(mode, axisPair, supportsDirectRx));
  }
}

[[nodiscard]] llvm::SmallVector<NativeGateKind, 4>
allowedGatesForEmitter(const SingleQubitEmitterSpec& emitter) {
  switch (emitter.mode) {
  case SingleQubitMode::ZSXX: {
    llvm::SmallVector<NativeGateKind, 4> gates{
        NativeGateKind::X, NativeGateKind::Sx, NativeGateKind::Rz};
    if (emitter.supportsDirectRx) {
      gates.push_back(NativeGateKind::Rx);
    }
    return gates;
  }
  case SingleQubitMode::U3:
    return {NativeGateKind::U};
  case SingleQubitMode::R:
    return {NativeGateKind::R};
  case SingleQubitMode::AxisPair:
    switch (emitter.axisPair) {
    case AxisPair::RxRz:
      return {NativeGateKind::Rx, NativeGateKind::Rz};
    case AxisPair::RxRy:
      return {NativeGateKind::Rx, NativeGateKind::Ry};
    case AxisPair::RyRz:
      return {NativeGateKind::Ry, NativeGateKind::Rz};
    }
    break;
  }
  llvm_unreachable("unknown single-qubit mode");
}

[[nodiscard]] llvm::SmallVector<NativeGateKind, 2>
allowedGatesForEntangler(EntanglerBasis entangler) {
  switch (entangler) {
  case EntanglerBasis::None:
    return {};
  case EntanglerBasis::Cx:
    return {NativeGateKind::Cx};
  case EntanglerBasis::Cz:
    return {NativeGateKind::Cz};
  }
  llvm_unreachable("unknown entangler basis");
}

void populateAllowedGates(NativeProfileSpec& spec) {
  spec.allowedGates.clear();
  for (const auto& emitter : spec.singleQubitEmitters) {
    const auto allowed = allowedGatesForEmitter(emitter);
    spec.allowedGates.insert(allowed.begin(), allowed.end());
  }
  for (const auto entangler : spec.entanglerBases) {
    const auto allowed = allowedGatesForEntangler(entangler);
    spec.allowedGates.insert(allowed.begin(), allowed.end());
  }
  if (spec.allowRzz) {
    spec.allowedGates.insert(NativeGateKind::Rzz);
  }
}

[[nodiscard]] std::optional<EntanglerBasis>
selectEntangler(const NativeProfileSpec& spec) {
  if (llvm::is_contained(spec.entanglerBases, EntanglerBasis::Cx)) {
    return EntanglerBasis::Cx;
  }
  if (llvm::is_contained(spec.entanglerBases, EntanglerBasis::Cz)) {
    return EntanglerBasis::Cz;
  }
  return std::nullopt;
}

[[nodiscard]] Matrix4x4 entanglerMatrix(EntanglerBasis entangler) {
  return entangler == EntanglerBasis::Cz ? twoQubitControlledZ()
                                         : twoQubitControlledX01();
}

[[nodiscard]] std::optional<TwoQubitNativeDecomposition>
decomposeWithEntangler(const Matrix4x4& target, EntanglerBasis entangler) {
  auto decomposer =
      TwoQubitBasisDecomposer::create(entanglerMatrix(entangler), 1.0);
  auto weyl = TwoQubitWeylDecomposition::create(target, std::nullopt);
  return decomposer.twoQubitDecompose(weyl, std::nullopt);
}

void emitGPhaseIfNonTrivial(OpBuilder& builder, Location loc, double phase) {
  constexpr double epsilon = 1e-12;
  if (std::abs(phase) > epsilon) {
    GPhaseOp::create(builder, loc, phase);
  }
}

[[nodiscard]] Value emitSingleQubitMatrix(OpBuilder& builder, Location loc,
                                          Value inQubit,
                                          const Matrix2x2& matrix,
                                          EulerBasis basis) {
  return *synthesizeUnitary1QEuler(builder, loc, inQubit, matrix,
                                   /*runSize=*/0, /*hasNonBasisGate=*/true,
                                   basis);
}

} // namespace

EulerBasis emitterEulerBasis(const SingleQubitEmitterSpec& emitter) {
  switch (emitter.mode) {
  case SingleQubitMode::ZSXX:
    return EulerBasis::ZSXX;
  case SingleQubitMode::U3:
    return EulerBasis::U;
  case SingleQubitMode::R:
    return EulerBasis::R;
  case SingleQubitMode::AxisPair:
    switch (emitter.axisPair) {
    case AxisPair::RxRz:
      return EulerBasis::XZX;
    case AxisPair::RxRy:
      return EulerBasis::XYX;
    case AxisPair::RyRz:
      return EulerBasis::ZYZ;
    }
    break;
  }
  llvm_unreachable("unknown single-qubit mode");
}

std::optional<NativeProfileSpec> parseNativeSpec(llvm::StringRef nativeGates) {
  const auto gates = parseGateSet(nativeGates);
  if (!gates || gates->empty()) {
    return std::nullopt;
  }
  const auto has = [&](NativeGateKind kind) { return gates->contains(kind); };

  NativeProfileSpec spec;

  if (has(NativeGateKind::U)) {
    addEmitterIfAbsent(spec.singleQubitEmitters, SingleQubitMode::U3);
  }
  const bool hasXSxRz = has(NativeGateKind::X) && has(NativeGateKind::Sx) &&
                        has(NativeGateKind::Rz);
  if (hasXSxRz) {
    addEmitterIfAbsent(spec.singleQubitEmitters, SingleQubitMode::ZSXX,
                       AxisPair::RxRz,
                       /*supportsDirectRx=*/has(NativeGateKind::Rx));
  }
  if (has(NativeGateKind::R)) {
    addEmitterIfAbsent(spec.singleQubitEmitters, SingleQubitMode::R);
  }
  struct AxisPairRule {
    AxisPair axis;
    NativeGateKind left;
    NativeGateKind right;
  };
  for (const auto& rule : {
           AxisPairRule{.axis = AxisPair::RxRz,
                        .left = NativeGateKind::Rx,
                        .right = NativeGateKind::Rz},
           AxisPairRule{.axis = AxisPair::RxRy,
                        .left = NativeGateKind::Rx,
                        .right = NativeGateKind::Ry},
           AxisPairRule{.axis = AxisPair::RyRz,
                        .left = NativeGateKind::Ry,
                        .right = NativeGateKind::Rz},
       }) {
    if (has(rule.left) && has(rule.right)) {
      addEmitterIfAbsent(spec.singleQubitEmitters, SingleQubitMode::AxisPair,
                         rule.axis);
    }
  }
  if (spec.singleQubitEmitters.empty()) {
    return std::nullopt;
  }

  if (has(NativeGateKind::Cx)) {
    spec.entanglerBases.push_back(EntanglerBasis::Cx);
  }
  if (has(NativeGateKind::Cz)) {
    spec.entanglerBases.push_back(EntanglerBasis::Cz);
  }
  spec.allowRzz = has(NativeGateKind::Rzz);

  populateAllowedGates(spec);
  return spec;
}

LogicalResult synthesizeUnitary2QWeyl(OpBuilder& builder, Location loc,
                                      Value qubit0, Value qubit1,
                                      const Matrix4x4& target,
                                      const NativeProfileSpec& spec,
                                      Value& outQubit0, Value& outQubit1) {
  const auto entangler = selectEntangler(spec);
  if (!entangler) {
    return failure();
  }
  const auto native = decomposeWithEntangler(target, *entangler);
  if (!native) {
    return failure();
  }
  const auto basis = emitterEulerBasis(spec.singleQubitEmitters.front());

  emitGPhaseIfNonTrivial(builder, loc, native->globalPhase);

  Value wire0 = qubit0;
  Value wire1 = qubit1;
  const auto& factors = native->singleQubitFactors;
  const std::uint8_t numBasisUses = native->numBasisUses;
  const auto emitFactor = [&](Value& wire, std::size_t index) {
    wire = emitSingleQubitMatrix(builder, loc, wire, factors[index], basis);
  };
  const auto emitEntangler = [&]() {
    auto ctrlOp = CtrlOp::create(
        builder, loc, ValueRange{wire0}, ValueRange{wire1},
        [&](ValueRange targetArgs) -> llvm::SmallVector<Value> {
          if (*entangler == EntanglerBasis::Cz) {
            return {ZOp::create(builder, loc, targetArgs[0]).getOutputQubit(0)};
          }
          return {XOp::create(builder, loc, targetArgs[0]).getOutputQubit(0)};
        });
    wire0 = ctrlOp.getOutputControl(0);
    wire1 = ctrlOp.getOutputTarget(0);
  };

  for (std::uint8_t i = 0; i < numBasisUses; ++i) {
    emitFactor(wire1, static_cast<std::size_t>(2 * i));
    emitFactor(wire0, static_cast<std::size_t>((2 * i) + 1));
    emitEntangler();
  }
  emitFactor(wire1, static_cast<std::size_t>(2 * numBasisUses));
  emitFactor(wire0, static_cast<std::size_t>((2 * numBasisUses) + 1));

  outQubit0 = wire0;
  outQubit1 = wire1;
  return success();
}

std::optional<std::uint8_t>
twoQubitEntanglerCount(const Matrix4x4& target, const NativeProfileSpec& spec) {
  const auto entangler = selectEntangler(spec);
  if (!entangler) {
    return std::nullopt;
  }
  const auto native = decomposeWithEntangler(target, *entangler);
  if (!native) {
    return std::nullopt;
  }
  return native->numBasisUses;
}

std::optional<NativeGateKind> nativeGateKindFor(UnitaryOpInterface op) {
  Operation* raw = op.getOperation();
  if (llvm::isa<UOp>(raw)) {
    return NativeGateKind::U;
  }
  if (llvm::isa<XOp>(raw)) {
    return NativeGateKind::X;
  }
  if (llvm::isa<SXOp>(raw)) {
    return NativeGateKind::Sx;
  }
  if (llvm::isa<RZOp, POp>(raw)) {
    return NativeGateKind::Rz;
  }
  if (llvm::isa<RXOp>(raw)) {
    return NativeGateKind::Rx;
  }
  if (llvm::isa<RYOp>(raw)) {
    return NativeGateKind::Ry;
  }
  if (llvm::isa<ROp>(raw)) {
    return NativeGateKind::R;
  }
  return std::nullopt;
}

bool allowsSingleQubitOp(UnitaryOpInterface op, const NativeProfileSpec& spec) {
  if (llvm::isa<BarrierOp, GPhaseOp>(op.getOperation())) {
    return true;
  }
  const auto gate = nativeGateKindFor(op);
  return gate && spec.allowedGates.contains(*gate);
}

std::optional<EntanglerBasis> entanglerBasisForSingleTargetCtrl(CtrlOp ctrl) {
  if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
    return std::nullopt;
  }
  Operation* body = ctrl.getBodyUnitary(0).getOperation();
  if (llvm::isa<XOp>(body)) {
    return EntanglerBasis::Cx;
  }
  if (llvm::isa<ZOp>(body)) {
    return EntanglerBasis::Cz;
  }
  return std::nullopt;
}

bool profileAllowsEntangler(const NativeProfileSpec& spec,
                            EntanglerBasis basis) {
  return llvm::is_contained(spec.entanglerBases, basis);
}

bool allowsSingleTargetCtrl(CtrlOp ctrl, const NativeProfileSpec& spec) {
  const auto basis = entanglerBasisForSingleTargetCtrl(ctrl);
  return basis && profileAllowsEntangler(spec, *basis);
}

bool allowsBareTwoQubitOp(Operation* op, const NativeProfileSpec& spec) {
  return spec.allowRzz && llvm::isa<RZZOp>(op) &&
         spec.allowedGates.contains(NativeGateKind::Rzz);
}

bool allowsTwoQubitOp(Operation* op, const NativeProfileSpec& spec) {
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
    return allowsSingleTargetCtrl(ctrl, spec);
  }
  return allowsBareTwoQubitOp(op, spec);
}

bool assignTwoQubitOpMatrix(Operation* op, Matrix4x4& matrix) {
  if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
    return false;
  }
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
    const auto basis = entanglerBasisForSingleTargetCtrl(ctrl);
    if (!basis) {
      return false;
    }
    matrix = *basis == EntanglerBasis::Cz ? twoQubitControlledZ()
                                          : twoQubitControlledX01();
    return true;
  }
  auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
  if (!unitary || !unitary.isTwoQubit()) {
    return false;
  }
  return unitary.getUnitaryMatrix4x4(matrix);
}

} // namespace mlir::qco::decomposition
