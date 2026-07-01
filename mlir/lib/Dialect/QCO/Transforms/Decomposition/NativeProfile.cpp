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
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace {

using mlir::qco::Matrix4x4;

constexpr Matrix4x4 kCanonicalControlledX =
    Matrix4x4::fromElements(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 1.0, 0.0, 0.0, 1.0, 0.0);

constexpr Matrix4x4 kCanonicalControlledZ =
    Matrix4x4::fromElements(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                            1.0, 0.0, 0.0, 0.0, 0.0, -1.0);

} // namespace

namespace mlir::qco::decomposition {

static std::optional<NativeGateKind> parseGateToken(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<NativeGateKind>>(name)
      .Case("u", NativeGateKind::U)
      .Case("x", NativeGateKind::X)
      .Case("sx", NativeGateKind::SX)
      .Cases("rz", "p", NativeGateKind::RZ)
      .Case("rx", NativeGateKind::RX)
      .Case("ry", NativeGateKind::RY)
      .Case("r", NativeGateKind::R)
      .Case("cx", NativeGateKind::CX)
      .Case("cz", NativeGateKind::CZ)
      .Case("rzz", NativeGateKind::RZZ)
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

/**
 * @brief Resolves the preferred single-qubit Euler basis for a parsed menu.
 *
 * Returns `std::nullopt` when no supported single-qubit synthesis strategy is
 * present. Priority matches @ref NativeProfileSpec::eulerBasis.
 */
[[nodiscard]] static std::optional<EulerBasis>
resolveEulerBasis(const llvm::DenseSet<NativeGateKind>& gates) {
  const auto has = [&](NativeGateKind kind) { return gates.contains(kind); };
  if (has(NativeGateKind::U)) {
    return EulerBasis::U;
  }
  if (has(NativeGateKind::X) && has(NativeGateKind::SX) &&
      has(NativeGateKind::RZ)) {
    return EulerBasis::ZSXX;
  }
  if (has(NativeGateKind::R)) {
    return EulerBasis::R;
  }
  if (has(NativeGateKind::RX) && has(NativeGateKind::RZ)) {
    return EulerBasis::XZX;
  }
  if (has(NativeGateKind::RX) && has(NativeGateKind::RY)) {
    return EulerBasis::XYX;
  }
  if (has(NativeGateKind::RY) && has(NativeGateKind::RZ)) {
    return EulerBasis::ZYZ;
  }
  return std::nullopt;
}

/**
 * @brief Picks the two-qubit entangler for Weyl synthesis.
 *
 * When both `cx` and `cz` appear in the menu, `cx` is preferred.
 */
[[nodiscard]] static std::optional<NativeGateKind>
selectEntangler(const llvm::DenseSet<NativeGateKind>& gates) {
  if (gates.contains(NativeGateKind::CX)) {
    return NativeGateKind::CX;
  }
  if (gates.contains(NativeGateKind::CZ)) {
    return NativeGateKind::CZ;
  }
  return std::nullopt;
}

static const TwoQubitBasisDecomposer&
cachedBasisDecomposer(NativeGateKind entangler) {
  switch (entangler) {
  case NativeGateKind::CX: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(kCanonicalControlledX, 1.0);
    return DECOMPOSER;
  }
  case NativeGateKind::CZ: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(kCanonicalControlledZ, 1.0);
    return DECOMPOSER;
  }
  default:
    llvm_unreachable("only CX/CZ are valid entanglers");
  }
}

static std::optional<NativeGateKind> gateKindFor(UnitaryOpInterface op) {
  return TypeSwitch<Operation*, std::optional<NativeGateKind>>(
             op.getOperation())
      .Case<UOp>([](UOp) { return NativeGateKind::U; })
      .Case<XOp>([](XOp) { return NativeGateKind::X; })
      .Case<SXOp>([](SXOp) { return NativeGateKind::SX; })
      .Case<RZOp, POp>([](auto) { return NativeGateKind::RZ; })
      .Case<RXOp>([](RXOp) { return NativeGateKind::RX; })
      .Case<RYOp>([](RYOp) { return NativeGateKind::RY; })
      .Case<ROp>([](ROp) { return NativeGateKind::R; })
      .Default([](Operation*) { return std::nullopt; });
}

static std::optional<NativeGateKind> entanglerKindFor(CtrlOp ctrl) {
  if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
    return std::nullopt;
  }
  auto bodyUnitary =
      utils::getSoleBodyUnitary<UnitaryOpInterface>(*ctrl.getBody());
  if (!bodyUnitary) {
    return std::nullopt;
  }
  return TypeSwitch<Operation*, std::optional<NativeGateKind>>(
             bodyUnitary.getOperation())
      .Case<XOp>([](XOp) { return NativeGateKind::CX; })
      .Case<ZOp>([](ZOp) { return NativeGateKind::CZ; })
      .Default([](Operation*) { return std::nullopt; });
}

EulerBasis NativeProfileSpec::eulerBasis() const {
  // Valid only for specs returned by @ref parseNativeSpec.
  return *resolveEulerBasis(gates);
}

std::optional<NativeProfileSpec> parseNativeSpec(llvm::StringRef nativeGates) {
  auto gates = parseGateSet(nativeGates);
  if (!gates || !resolveEulerBasis(*gates) || !selectEntangler(*gates)) {
    return std::nullopt;
  }
  return NativeProfileSpec{.gates = std::move(*gates)};
}

LogicalResult synthesizeUnitary2QWeyl(OpBuilder& builder, Location loc,
                                      Value qubit0, Value qubit1,
                                      const Matrix4x4& target,
                                      const NativeProfileSpec& spec,
                                      Value& outQubit0, Value& outQubit1) {
  const auto entangler = selectEntangler(spec.gates);
  if (!entangler) {
    return failure();
  }
  const auto native = cachedBasisDecomposer(*entangler).decomposeTarget(target);
  if (!native) {
    return failure();
  }

  emitGPhaseIfNeeded(builder, loc, native->globalPhase);

  Value wire0 = qubit0;
  Value wire1 = qubit1;
  const auto& factors = native->singleQubitFactors;
  const std::uint8_t numBasisUses = native->numBasisUses;
  const EulerBasis basis = spec.eulerBasis();
  const bool emitCz = (*entangler == NativeGateKind::CZ);
  const auto emitFactor = [&](Value& wire, std::size_t index) {
    const auto synthesized = synthesizeUnitary1QEuler(
        builder, loc, wire, factors[index], /*runSize=*/0,
        /*hasNonBasisGate=*/true, basis);
    if (!synthesized) {
      llvm_unreachable("forced full synthesis must succeed");
    }
    wire = *synthesized;
  };
  const auto emitEntangler = [&]() {
    auto ctrlOp = CtrlOp::create(
        builder, loc, ValueRange{wire0}, ValueRange{wire1},
        [&](ValueRange targetArgs) -> SmallVector<Value> {
          if (emitCz) {
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
  const auto entangler = selectEntangler(spec.gates);
  if (!entangler) {
    return std::nullopt;
  }
  const auto native = cachedBasisDecomposer(*entangler).decomposeTarget(target);
  if (!native) {
    return std::nullopt;
  }
  return native->numBasisUses;
}

bool allowsOp(Operation* op, const NativeProfileSpec& spec) {
  return TypeSwitch<Operation*, bool>(op)
      .Case<BarrierOp, GPhaseOp>([](auto) { return true; })
      .Case<CtrlOp>([&](CtrlOp ctrl) {
        const auto kind = entanglerKindFor(ctrl);
        return kind && spec.gates.contains(*kind);
      })
      .Case<RZZOp>(
          [&](RZZOp) { return spec.gates.contains(NativeGateKind::RZZ); })
      .Case<UnitaryOpInterface>([&](UnitaryOpInterface unitary) {
        if (!unitary.isSingleQubit()) {
          return false;
        }
        const auto gate = gateKindFor(unitary);
        return gate && spec.gates.contains(*gate);
      })
      .Default([](Operation*) { return false; });
}

} // namespace mlir::qco::decomposition
