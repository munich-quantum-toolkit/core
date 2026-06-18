/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>

#include <cstdint>

/// Types for native gate synthesis: the resolved menu and its emitters.

namespace mlir::qco::native_synth {

/// Two-axis token pairs (`rx`+`rz`, `rx`+`ry`, `ry`+`rz`) that can be selected
/// as the single-qubit menu in a `NativeProfileSpec`.
enum class AxisPair : std::uint8_t { RxRz, RxRy, RyRz };

/// Single-qubit emission strategy.
enum class SingleQubitMode : std::uint8_t {
  /// Emit `{X, Sx, Rz}` via the ZSXX Euler decomposition. When the spec's
  /// `supportsDirectRx` is set, the emitter additionally passes Rx through
  /// unchanged and expands Ry / R via an `rz * rx * rz` sandwich.
  ZSXX,
  /// Emit a single `u(theta, phi, lambda)` op.
  U3,
  /// Emit `R(theta, phi)` via the XYX Euler decomposition.
  R,
  /// Emit one of the three two-axis rotation pairs selected by `axisPair`.
  AxisPair,
};

/// Two-qubit entangling basis selected by a profile.
enum class EntanglerBasis : std::uint8_t { None, Cx, Cz };

/// Profile-level classification of a native gate. Used both to describe the
/// menu (`NativeProfileSpec::allowedGates`) and to classify already-lowered
/// output ops in policy checks.
enum class NativeGateKind : std::uint8_t {
  U,
  X,
  Sx,
  Rz,
  Rx,
  Ry,
  R,
  Cx,
  Cz,
  Rzz,
};

/// Single-qubit emitter specification: the target mode plus any modifiers
/// (axis pair, whether direct Rx emission is permitted).
struct SingleQubitEmitterSpec {
  SingleQubitMode mode = SingleQubitMode::U3;
  AxisPair axisPair = AxisPair::RxRz;
  /// Only meaningful for `SingleQubitMode::ZSXX`: when set, the emitter may
  /// emit Rx / Ry / R directly (via an `rz * rx * rz` sandwich for the latter
  /// two) instead of falling back to the ZSXX Euler sequence.
  bool supportsDirectRx = false;
};

/// Resolved menu: emitters to try for 1q synthesis and entangler bases for 2q.
/// Built by `resolveNativeGatesSpec`. Single-qubit synthesis is deterministic:
/// the first emitter is preferred and its Euler basis drives matrix synthesis.
struct NativeProfileSpec {
  bool allowRzz = false;
  llvm::DenseSet<NativeGateKind> allowedGates;
  llvm::SmallVector<SingleQubitEmitterSpec> singleQubitEmitters;
  llvm::SmallVector<EntanglerBasis> entanglerBases;
};

} // namespace mlir::qco::native_synth
