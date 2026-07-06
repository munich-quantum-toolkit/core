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

constexpr Matrix4x4 CANONICAL_CONTROLLED_X =
    Matrix4x4::fromElements(1.0, 0.0, 0.0, 0.0,  // row 0
                            0.0, 1.0, 0.0, 0.0,  // row 1
                            0.0, 0.0, 0.0, 1.0,  // row 2
                            0.0, 0.0, 1.0, 0.0); // row 3

constexpr Matrix4x4 CANONICAL_CONTROLLED_Z =
    Matrix4x4::fromDiagonal(1., 1., 1., -1.);

} // namespace

namespace mlir::qco::decomposition {

static std::optional<NativeGateKind> parseGateToken(StringRef name) {
  return StringSwitch<std::optional<NativeGateKind>>(name)
      .Case("u", NativeGateKind::U)
      .Case("x", NativeGateKind::X)
      .Case("sx", NativeGateKind::SX)
      .Cases({"rz", "p"}, NativeGateKind::RZ)
      .Case("rx", NativeGateKind::RX)
      .Case("ry", NativeGateKind::RY)
      .Case("r", NativeGateKind::R)
      .Case("cx", NativeGateKind::CX)
      .Case("cz", NativeGateKind::CZ)
      .Default(std::nullopt);
}

static std::optional<DenseSet<NativeGateKind>>
parseGateSet(StringRef nativeGates) {
  DenseSet<NativeGateKind> gates;
  SmallVector<StringRef> parts;
  nativeGates.split(parts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef part : parts) {
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
 * @brief Resolves the preferred single-qubit Euler basis for a parsed gateset.
 *
 * Returns `std::nullopt` when no supported single-qubit synthesis strategy is
 * present. Priority matches @ref NativeProfileSpec::eulerBasis.
 */
[[nodiscard]] static std::optional<EulerBasis>
resolveEulerBasis(const DenseSet<NativeGateKind>& gates) {
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
 * Only `cx` and `cz` are supported by @ref TwoQubitBasisDecomposer. When both
 * appear in the gateset, `cx` is preferred.
 */
[[nodiscard]] static std::optional<NativeGateKind>
selectEntangler(const DenseSet<NativeGateKind>& gates) {
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
        TwoQubitBasisDecomposer::create(CANONICAL_CONTROLLED_X, 1.0);
    return DECOMPOSER;
  }
  case NativeGateKind::CZ: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(CANONICAL_CONTROLLED_Z, 1.0);
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
  if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1 ||
      ctrl.getNumBodyUnitaries() != 1) {
    return std::nullopt;
  }
  return TypeSwitch<Operation*, std::optional<NativeGateKind>>(
             ctrl.getBodyUnitary(0).getOperation())
      .Case<XOp>([](XOp) { return NativeGateKind::CX; })
      .Case<ZOp>([](ZOp) { return NativeGateKind::CZ; })
      .Default([](Operation*) { return std::nullopt; });
}

EulerBasis NativeProfileSpec::eulerBasis() const {
  // Valid only for specs returned by @ref NativeProfileSpec::parse.
  return *resolveEulerBasis(gates);
}

std::optional<NativeProfileSpec>
NativeProfileSpec::parse(StringRef nativeGates) {
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
  const auto basis = resolveEulerBasis(spec.gates);
  if (!basis) {
    return failure();
  }

  emitGPhaseIfNeeded(builder, loc, native->globalPhase);

  Value wire0 = qubit0;
  Value wire1 = qubit1;
  const auto& factors = native->singleQubitFactors;
  const std::uint8_t numBasisUses = native->numBasisUses;
  const bool emitCz = (*entangler == NativeGateKind::CZ);
  const auto emitFactor = [&](Value& wire, std::size_t index) {
    const auto synthesized = synthesizeUnitary1QEuler(
        builder, loc, wire, factors[index], /*runSize=*/0,
        /*hasNonBasisGate=*/true, *basis);
    if (!synthesized) {
      llvm_unreachable("forced full synthesis must succeed");
    }
    wire = *synthesized;
  };
  const auto emitEntangler = [&]() {
    auto ctrlOp = CtrlOp::create(
        builder, loc, wire0, wire1,
        [&](ValueRange targetArgs) -> SmallVector<Value> {
          if (emitCz) {
            return {ZOp::create(builder, loc, targetArgs[0]).getOutputQubit(0)};
          }
          return {XOp::create(builder, loc, targetArgs[0]).getOutputQubit(0)};
        });
    wire0 = ctrlOp.getOutputControl(0);
    wire1 = ctrlOp.getOutputTarget(0);
  };

  for (std::uint8_t layer = 0; layer <= numBasisUses; ++layer) {
    emitFactor(wire1, static_cast<std::size_t>(2 * layer));
    emitFactor(wire0, static_cast<std::size_t>((2 * layer) + 1));
    if (layer < numBasisUses) {
      emitEntangler();
    }
  }

  outQubit0 = wire0;
  outQubit1 = wire1;
  return success();
}

std::optional<std::uint8_t>
NativeProfileSpec::twoQubitEntanglerCount(const Matrix4x4& target) const {
  const auto entangler = selectEntangler(gates);
  if (!entangler) {
    return std::nullopt;
  }
  const auto native = cachedBasisDecomposer(*entangler).decomposeTarget(target);
  if (!native) {
    return std::nullopt;
  }
  return native->numBasisUses;
}

bool NativeProfileSpec::allowsOp(Operation* op) const {
  return TypeSwitch<Operation*, bool>(op)
      .Case<BarrierOp, GPhaseOp>([](auto) { return true; })
      .Case<CtrlOp>([&](CtrlOp ctrl) {
        const auto kind = entanglerKindFor(ctrl);
        return kind && gates.contains(*kind);
      })
      .Case<UnitaryOpInterface>([&](UnitaryOpInterface unitary) {
        if (!unitary.isSingleQubit()) {
          return false;
        }
        const auto gate = gateKindFor(unitary);
        return gate && gates.contains(*gate);
      })
      .Default([](Operation*) { return false; });
}

} // namespace mlir::qco::decomposition
