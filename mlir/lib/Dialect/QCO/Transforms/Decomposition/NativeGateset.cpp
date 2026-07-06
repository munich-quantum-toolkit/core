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

#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Support/LLVM.h>

#include <optional>
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

std::optional<TwoQubitNativeDecomposition>
NativeGateset::decomposeTarget(const Matrix4x4& target) const {
  if (!entangler) {
    return std::nullopt;
  }
  return cachedNativeBasisDecomposer(*entangler).decomposeTarget(target);
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

} // namespace mlir::qco::decomposition
