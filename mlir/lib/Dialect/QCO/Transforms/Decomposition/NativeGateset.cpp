/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/NativeGateset.h"

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

#include <numbers>
#include <optional>
#include <string>
#include <utility>

namespace mlir::qco::decomposition {

static std::optional<NativeGateKind> parseGateToken(StringRef name) {
  return StringSwitch<std::optional<NativeGateKind>>(name)
      .Case("u", NativeGateKind::U)
      .Case("x", NativeGateKind::X)
      .Case("sx", NativeGateKind::SX)
      .Case("rz", NativeGateKind::RZ)
      .Case("rx", NativeGateKind::RX)
      .Case("ry", NativeGateKind::RY)
      .Case("r", NativeGateKind::R)
      .Case("rxx", NativeGateKind::RXX)
      .Case("ryy", NativeGateKind::RYY)
      .Case("rzx", NativeGateKind::RZX)
      .Case("rzz", NativeGateKind::RZZ)
      .Case("iswap", NativeGateKind::ISWAP)
      .Case("cz", NativeGateKind::CZ)
      .Case("cx", NativeGateKind::CX)
      .Case("ecr", NativeGateKind::ECR)
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
 * present. Cached on @ref NativeGateset by @ref NativeGateset::parse.
 */
[[nodiscard]] static std::optional<EulerBasis>
resolveEulerBasis(const DenseSet<NativeGateKind>& gates) {
  if (gates.contains(NativeGateKind::U)) {
    return EulerBasis::U;
  }
  if (gates.contains(NativeGateKind::X) && gates.contains(NativeGateKind::SX) &&
      gates.contains(NativeGateKind::RZ)) {
    return EulerBasis::ZSXX;
  }
  if (gates.contains(NativeGateKind::R)) {
    return EulerBasis::R;
  }
  if (gates.contains(NativeGateKind::RX) &&
      gates.contains(NativeGateKind::RZ)) {
    return EulerBasis::XZX;
  }
  if (gates.contains(NativeGateKind::RX) &&
      gates.contains(NativeGateKind::RY)) {
    return EulerBasis::XYX;
  }
  if (gates.contains(NativeGateKind::RY) &&
      gates.contains(NativeGateKind::RZ)) {
    return EulerBasis::ZYZ;
  }
  return std::nullopt;
}

/**
 * @brief Picks the two-qubit entangler for Weyl synthesis.
 *
 * When multiple entanglers appear in the gateset, preference is
 * **RXX > RYY > RZX > RZZ > iSWAP > CZ > CX > ECR**.
 */
[[nodiscard]] static std::optional<NativeGateKind>
selectEntangler(const DenseSet<NativeGateKind>& gates) {
  if (gates.contains(NativeGateKind::RXX)) {
    return NativeGateKind::RXX;
  }
  if (gates.contains(NativeGateKind::RYY)) {
    return NativeGateKind::RYY;
  }
  if (gates.contains(NativeGateKind::RZX)) {
    return NativeGateKind::RZX;
  }
  if (gates.contains(NativeGateKind::RZZ)) {
    return NativeGateKind::RZZ;
  }
  if (gates.contains(NativeGateKind::ISWAP)) {
    return NativeGateKind::ISWAP;
  }
  if (gates.contains(NativeGateKind::CZ)) {
    return NativeGateKind::CZ;
  }
  if (gates.contains(NativeGateKind::CX)) {
    return NativeGateKind::CX;
  }
  if (gates.contains(NativeGateKind::ECR)) {
    return NativeGateKind::ECR;
  }
  return std::nullopt;
}

static constexpr Matrix4x4 CANONICAL_CONTROLLED_X =
    Matrix4x4::fromElements(1.0, 0.0, 0.0, 0.0,  // row 0
                            0.0, 1.0, 0.0, 0.0,  // row 1
                            0.0, 0.0, 0.0, 1.0,  // row 2
                            0.0, 0.0, 1.0, 0.0); // row 3

static constexpr Matrix4x4 CANONICAL_CONTROLLED_Z =
    Matrix4x4::fromDiagonal(1., 1., 1., -1.);

static const TwoQubitBasisDecomposer&
cachedNativeBasisDecomposer(NativeGateKind entangler) {
  switch (entangler) {
  case NativeGateKind::RXX: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(
            RXXOp::unitaryMatrix(std::numbers::pi / 2.0), 1.0);
    return DECOMPOSER;
  }
  case NativeGateKind::RYY: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(
            RYYOp::unitaryMatrix(std::numbers::pi / 2.0), 1.0);
    return DECOMPOSER;
  }
  case NativeGateKind::RZX: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(
            RZXOp::unitaryMatrix(std::numbers::pi / 2.0), 1.0);
    return DECOMPOSER;
  }
  case NativeGateKind::RZZ: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(
            RZZOp::unitaryMatrix(std::numbers::pi / 2.0), 1.0);
    return DECOMPOSER;
  }
  case NativeGateKind::ISWAP: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(iSWAPOp::getUnitaryMatrix(), 1.0);
    return DECOMPOSER;
  }
  case NativeGateKind::CZ: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(CANONICAL_CONTROLLED_Z, 1.0);
    return DECOMPOSER;
  }
  case NativeGateKind::CX: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(CANONICAL_CONTROLLED_X, 1.0);
    return DECOMPOSER;
  }
  case NativeGateKind::ECR: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(ECROp::getUnitaryMatrix(), 1.0);
    return DECOMPOSER;
  }
  default:
    llvm_unreachable(
        "only RXX/RYY/RZX/RZZ/ISWAP/CZ/CX/ECR are valid entanglers");
  }
}

std::optional<TwoQubitNativeDecomposition>
NativeGateset::decomposeTarget(const Matrix4x4& target) const {
  if (!entangler) {
    return std::nullopt;
  }
  return cachedNativeBasisDecomposer(*entangler).decomposeTarget(target);
}

static std::optional<NativeGateKind> gateKindFor(UnitaryOpInterface op) {
  return TypeSwitch<Operation*, std::optional<NativeGateKind>>(
             op.getOperation())
      .Case<UOp>([](UOp) { return NativeGateKind::U; })
      .Case<XOp>([](XOp) { return NativeGateKind::X; })
      .Case<SXOp>([](SXOp) { return NativeGateKind::SX; })
      .Case<RZOp>([](RZOp) { return NativeGateKind::RZ; })
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

bool NativeGateset::allowsOp(Operation* op) const {
  return TypeSwitch<Operation*, bool>(op)
      .Case<BarrierOp, GPhaseOp>([](auto) { return true; })
      .Case<RXXOp>([&](RXXOp) { return gates.contains(NativeGateKind::RXX); })
      .Case<RYYOp>([&](RYYOp) { return gates.contains(NativeGateKind::RYY); })
      .Case<RZXOp>([&](RZXOp) { return gates.contains(NativeGateKind::RZX); })
      .Case<RZZOp>([&](RZZOp) { return gates.contains(NativeGateKind::RZZ); })
      .Case<iSWAPOp>(
          [&](iSWAPOp) { return gates.contains(NativeGateKind::ISWAP); })
      .Case<CtrlOp>([&](CtrlOp ctrl) {
        const auto kind = entanglerKindFor(ctrl);
        return kind && gates.contains(*kind);
      })
      .Case<ECROp>([&](ECROp) { return gates.contains(NativeGateKind::ECR); })
      .Case<UnitaryOpInterface>([&](UnitaryOpInterface unitary) {
        if (!unitary.isSingleQubit()) {
          return false;
        }
        const auto gate = gateKindFor(unitary);
        return gate && gates.contains(*gate);
      })
      .Default([](Operation*) { return false; });
}

std::optional<NativeGateset> NativeGateset::parse(StringRef nativeGates) {
  auto gates = parseGateSet(nativeGates);
  if (!gates) {
    return std::nullopt;
  }
  const auto euler = resolveEulerBasis(*gates);
  const auto entangler = selectEntangler(*gates);
  if (!euler || !entangler) {
    return std::nullopt;
  }
  return NativeGateset{
      .gates = std::move(*gates),
      .eulerBasis = euler,
      .entangler = entangler,
  };
}

static StringRef normalizeGateAlias(StringRef token) {
  token = token.trim();
  if (token.equals_insensitive("prx")) {
    return "r";
  }
  if (token.equals_insensitive("u3")) {
    return "u";
  }
  if (token.equals_insensitive("cnot")) {
    return "cx";
  }
  return token;
}

static void insertEulerConstituents(DenseSet<NativeGateKind>& selected,
                                    EulerBasis euler) {
  switch (euler) {
  case EulerBasis::U:
    selected.insert(NativeGateKind::U);
    break;
  case EulerBasis::ZSXX:
    selected.insert(NativeGateKind::X);
    selected.insert(NativeGateKind::SX);
    selected.insert(NativeGateKind::RZ);
    break;
  case EulerBasis::R:
    selected.insert(NativeGateKind::R);
    break;
  case EulerBasis::XZX:
    selected.insert(NativeGateKind::RX);
    selected.insert(NativeGateKind::RZ);
    break;
  case EulerBasis::XYX:
    selected.insert(NativeGateKind::RX);
    selected.insert(NativeGateKind::RY);
    break;
  case EulerBasis::ZYZ:
    selected.insert(NativeGateKind::RY);
    selected.insert(NativeGateKind::RZ);
    break;
  }
}

std::optional<NativeGateset>
NativeGateset::fromOperationNames(ArrayRef<StringRef> names) {
  DenseSet<NativeGateKind> recognized;
  for (StringRef name : names) {
    std::string lowered = name.trim().lower();
    if (lowered.empty()) {
      continue;
    }
    const StringRef token = normalizeGateAlias(lowered);
    const auto gate = parseGateToken(token);
    if (gate) {
      recognized.insert(*gate);
    }
  }
  const auto euler = resolveEulerBasis(recognized);
  const auto entangler = selectEntangler(recognized);
  if (!euler || !entangler) {
    return std::nullopt;
  }
  DenseSet<NativeGateKind> selected;
  insertEulerConstituents(selected, *euler);
  selected.insert(*entangler);
  return NativeGateset{
      .gates = std::move(selected),
      .eulerBasis = euler,
      .entangler = entangler,
  };
}

std::string NativeGateset::toMenuString() const {
  if (!eulerBasis || !entangler) {
    return {};
  }
  std::string out;
  auto append = [&](StringRef tok) {
    if (!out.empty()) {
      out.push_back(',');
    }
    out.append(tok.str());
  };
  switch (*eulerBasis) {
  case EulerBasis::U:
    append("u");
    break;
  case EulerBasis::ZSXX:
    append("x");
    append("sx");
    append("rz");
    break;
  case EulerBasis::R:
    append("r");
    break;
  case EulerBasis::XZX:
    append("rx");
    append("rz");
    break;
  case EulerBasis::XYX:
    append("rx");
    append("ry");
    break;
  case EulerBasis::ZYZ:
    append("ry");
    append("rz");
    break;
  }
  switch (*entangler) {
  case NativeGateKind::RXX:
    append("rxx");
    break;
  case NativeGateKind::RYY:
    append("ryy");
    break;
  case NativeGateKind::RZX:
    append("rzx");
    break;
  case NativeGateKind::RZZ:
    append("rzz");
    break;
  case NativeGateKind::ISWAP:
    append("iswap");
    break;
  case NativeGateKind::CZ:
    append("cz");
    break;
  case NativeGateKind::CX:
    append("cx");
    break;
  case NativeGateKind::ECR:
    append("ecr");
    break;
  default:
    llvm_unreachable(
        "only RXX/RYY/RZX/RZZ/ISWAP/CZ/CX/ECR are valid entanglers");
  }
  return out;
}

} // namespace mlir::qco::decomposition
