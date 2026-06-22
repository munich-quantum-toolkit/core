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

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

using mlir::qco::Matrix2x2;
using mlir::qco::Matrix4x4;

namespace mlir::qco::decomposition {

static std::optional<NativeGateKind> parseGateToken(llvm::StringRef name) {
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

static std::optional<llvm::DenseSet<NativeGateKind>>
parseGateSet(llvm::StringRef nativeGates) {
  llvm::DenseSet<NativeGateKind> gates;
  SmallVector<llvm::StringRef> parts;
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

static bool
hasSingleQubitStrategy(const llvm::DenseSet<NativeGateKind>& gates) {
  const auto has = [&](NativeGateKind kind) { return gates.contains(kind); };
  if (has(NativeGateKind::U)) {
    return true;
  }
  if (has(NativeGateKind::X) && has(NativeGateKind::Sx) &&
      has(NativeGateKind::Rz)) {
    return true;
  }
  if (has(NativeGateKind::R)) {
    return true;
  }
  return (has(NativeGateKind::Rx) && has(NativeGateKind::Rz)) ||
         (has(NativeGateKind::Rx) && has(NativeGateKind::Ry)) ||
         (has(NativeGateKind::Ry) && has(NativeGateKind::Rz));
}

static std::optional<NativeGateKind>
selectEntangler(const NativeProfileSpec& spec) {
  if (spec.gates.contains(NativeGateKind::Cx)) {
    return NativeGateKind::Cx;
  }
  if (spec.gates.contains(NativeGateKind::Cz)) {
    return NativeGateKind::Cz;
  }
  return std::nullopt;
}

static Matrix4x4 entanglerMatrix(NativeGateKind entangler) {
  return entangler == NativeGateKind::Cz ? mlir::qco::twoQubitControlledZ()
                                         : mlir::qco::twoQubitControlledX01();
}

static std::optional<TwoQubitNativeDecomposition>
decomposeForProfile(const Matrix4x4& target, const NativeProfileSpec& spec) {
  const auto entangler = selectEntangler(spec);
  if (!entangler) {
    return std::nullopt;
  }
  return decomposeTwoQubitWithBasis(target, entanglerMatrix(*entangler));
}

static void emitGPhaseIfNonTrivial(OpBuilder& builder, Location loc,
                                   double phase) {
  constexpr double epsilon = 1e-12;
  if (std::abs(phase) > epsilon) {
    GPhaseOp::create(builder, loc, phase);
  }
}

static Value emitSingleQubitMatrix(OpBuilder& builder, Location loc,
                                   Value inQubit, const Matrix2x2& matrix,
                                   EulerBasis basis) {
  return *synthesizeUnitary1QEuler(builder, loc, inQubit, matrix,
                                   /*runSize=*/0, /*hasNonBasisGate=*/true,
                                   basis);
}

static std::optional<NativeGateKind> gateKindFor(UnitaryOpInterface op) {
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

static std::optional<NativeGateKind> entanglerKindFor(CtrlOp ctrl) {
  if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
    return std::nullopt;
  }
  Operation* body = ctrl.getBodyUnitary(0).getOperation();
  if (llvm::isa<XOp>(body)) {
    return NativeGateKind::Cx;
  }
  if (llvm::isa<ZOp>(body)) {
    return NativeGateKind::Cz;
  }
  return std::nullopt;
}

EulerBasis NativeProfileSpec::eulerBasis() const {
  const auto has = [&](NativeGateKind kind) { return gates.contains(kind); };
  if (has(NativeGateKind::U)) {
    return EulerBasis::U;
  }
  if (has(NativeGateKind::X) && has(NativeGateKind::Sx) &&
      has(NativeGateKind::Rz)) {
    return EulerBasis::ZSXX;
  }
  if (has(NativeGateKind::R)) {
    return EulerBasis::R;
  }
  if (has(NativeGateKind::Rx) && has(NativeGateKind::Rz)) {
    return EulerBasis::XZX;
  }
  if (has(NativeGateKind::Rx) && has(NativeGateKind::Ry)) {
    return EulerBasis::XYX;
  }
  if (has(NativeGateKind::Ry) && has(NativeGateKind::Rz)) {
    return EulerBasis::ZYZ;
  }
  llvm_unreachable("parseNativeSpec guarantees a synthesizable basis");
}

std::optional<NativeProfileSpec> parseNativeSpec(llvm::StringRef nativeGates) {
  auto gates = parseGateSet(nativeGates);
  if (!gates || gates->empty() || !hasSingleQubitStrategy(*gates)) {
    return std::nullopt;
  }
  return NativeProfileSpec{.gates = std::move(gates).value()};
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
  const auto native = decomposeForProfile(target, spec);
  if (!native) {
    return failure();
  }
  const auto basis = spec.eulerBasis();

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
        [&](ValueRange targetArgs) -> SmallVector<Value> {
          if (*entangler == NativeGateKind::Cz) {
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
  const auto native = decomposeForProfile(target, spec);
  if (!native) {
    return std::nullopt;
  }
  return native->numBasisUses;
}

bool allowsOp(Operation* op, const NativeProfileSpec& spec) {
  if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
    return true;
  }
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
    const auto kind = entanglerKindFor(ctrl);
    return kind && spec.gates.contains(*kind);
  }
  if (llvm::isa<RZZOp>(op)) {
    return spec.gates.contains(NativeGateKind::Rzz);
  }
  auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
  if (!unitary || !unitary.isSingleQubit()) {
    return false;
  }
  const auto gate = gateKindFor(unitary);
  return gate && spec.gates.contains(*gate);
}

} // namespace mlir::qco::decomposition
