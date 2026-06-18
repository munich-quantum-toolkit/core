/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/NativeSpec.h"

#include "mlir/Dialect/QCO/Transforms/Decomposition/EulerBasis.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/ErrorHandling.h>

#include <optional>
#include <utility>

namespace mlir::qco::native_synth {

/// Map a single native-gate token (lower-case, no whitespace) to its
/// `NativeGateKind`.
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

/// Parse a comma-separated native-gate menu (e.g. `"u,cx,rzz"`) into the set
/// of `NativeGateKind`s it names.
static std::optional<llvm::DenseSet<NativeGateKind>>
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

/// Build a fully-resolved `SingleQubitEmitterSpec` for `mode`, including the
/// list of Euler bases the matrix-fallback path is allowed to use.
static SingleQubitEmitterSpec
makeEmitterSpec(SingleQubitMode mode, AxisPair axisPair = AxisPair::RxRz,
                bool supportsDirectRx = false) {
  llvm::SmallVector<decomposition::GateEulerBasis> bases;
  switch (mode) {
  case SingleQubitMode::ZSXX:
    bases = {decomposition::GateEulerBasis::ZSXX};
    break;
  case SingleQubitMode::U3:
    bases = {decomposition::GateEulerBasis::U3};
    break;
  case SingleQubitMode::R:
    // XYX decomposes any 1Q unitary into Rx-Ry-Rx chains, all of which the
    // R emitter lowers back into the native R(theta, phi) gate.
    bases = {decomposition::GateEulerBasis::XYX};
    break;
  case SingleQubitMode::AxisPair:
    bases = getEulerBasesForAxisPair(axisPair);
    break;
  }
  return {.mode = mode,
          .axisPair = axisPair,
          .eulerBases = std::move(bases),
          .supportsDirectRx = supportsDirectRx};
}

/// Append a new emitter for `(mode, axisPair, supportsDirectRx)` to
/// `emitters` iff no equivalent entry is already present.
static void
addEmitterIfAbsent(llvm::SmallVectorImpl<SingleQubitEmitterSpec>& emitters,
                   SingleQubitMode mode, AxisPair axisPair = AxisPair::RxRz,
                   bool supportsDirectRx = false) {
  const bool present = llvm::any_of(emitters, [&](const auto& e) {
    return e.mode == mode && e.axisPair == axisPair &&
           e.supportsDirectRx == supportsDirectRx;
  });
  if (!present) {
    emitters.push_back(makeEmitterSpec(mode, axisPair, supportsDirectRx));
  }
}

/// Enumerate the native gate kinds that `emitter` may actually emit.
static llvm::SmallVector<NativeGateKind, 4>
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

/// Enumerate the native entangling gate kinds that `entangler` may emit.
static llvm::SmallVector<NativeGateKind, 2>
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

/// Rebuild `spec.allowedGates` as the union of the gate kinds produced by
/// every resolved emitter, entangler, and (optionally) `Rzz`.
static void populateAllowedGates(NativeProfileSpec& spec) {
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

llvm::SmallVector<decomposition::GateEulerBasis>
getEulerBasesForAxisPair(AxisPair axisPair) {
  switch (axisPair) {
  case AxisPair::RxRz:
    return {decomposition::GateEulerBasis::XZX};
  case AxisPair::RxRy:
    return {decomposition::GateEulerBasis::XYX};
  case AxisPair::RyRz:
    return {decomposition::GateEulerBasis::ZYZ};
  }
  llvm_unreachable("unknown axis pair");
}

std::optional<NativeProfileSpec>
resolveNativeGatesSpec(llvm::StringRef nativeGates) {
  const auto gates = parseGateSet(nativeGates);
  if (!gates || gates->empty()) {
    return std::nullopt;
  }
  const auto has = [&](NativeGateKind kind) { return gates->contains(kind); };

  NativeProfileSpec spec;

  // Derive all legal single-qubit emitters from the declared menu. Each
  // emitter mode requires the *conjunction* of its constituent gate kinds
  // to be on the menu -- for example, ZSXX needs X, Sx, and Rz all present,
  // because the decomposer unconditionally emits all three. `supportsDirectRx`
  // is an independent capability that enables a fast-path for `Rx(theta)`
  // inputs when `Rx` is additionally available, but ZSXX itself does not
  // depend on `Rx`.
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

} // namespace mlir::qco::native_synth
