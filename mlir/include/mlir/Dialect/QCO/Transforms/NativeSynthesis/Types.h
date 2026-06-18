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

#include "mlir/Dialect/QCO/Transforms/Decomposition/EulerBasis.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateSequence.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>

#include <cstdint>
#include <memory>

/// Types for native gate synthesis: menu, emitters, candidates, score weights.

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
/// (axis pair, Euler bases to consider when decomposing, whether direct Rx
/// emission is permitted).
struct SingleQubitEmitterSpec {
  SingleQubitMode mode = SingleQubitMode::U3;
  AxisPair axisPair = AxisPair::RxRz;
  llvm::SmallVector<decomposition::GateEulerBasis> eulerBases;
  /// Only meaningful for `SingleQubitMode::ZSXX`: when set, the emitter may
  /// emit Rx / Ry / R directly (via an `rz * rx * rz` sandwich for the latter
  /// two) instead of falling back to the ZSXX Euler sequence.
  bool supportsDirectRx = false;
};

/// Resolved menu: emitters to try for 1q synthesis and entangler bases for 2q.
/// Built by `resolveNativeGatesSpec`.
struct NativeProfileSpec {
  bool allowRzz = false;
  llvm::DenseSet<NativeGateKind> allowedGates;
  llvm::SmallVector<SingleQubitEmitterSpec> singleQubitEmitters;
  llvm::SmallVector<EntanglerBasis> entanglerBases;
};

/// Weights for the deterministic local cost model. Candidate cost is
/// `twoQ * #2q + oneQ * #1q + depth * localDepth`; lower is better.
struct ScoreWeights {
  double twoQ = 1.0;
  double oneQ = 0.1;
  double depth = 0.01;
};

/// Gate counts describing a synthesized candidate.
struct CandidateMetrics {
  unsigned numOneQ = 0;
  unsigned numTwoQ = 0;
  unsigned depth = 0;
};

/// Tie-break classes in preference order (lower wins). Used as the final
/// structural tiebreaker in `isBetterScore` after the weighted cost and the
/// raw 2q/depth/1q counts.
enum class CandidateClass : std::uint8_t {
  NativePassthrough = 0,
  DirectSingleQ = 1,
  MatrixSingleQ = 2,
  TwoQubitBasisRewrite = 3,
  XxPlusMinusViaRzz = 4,
};

/// Generic candidate wrapper carrying a typed rewrite plan payload.
template <typename Payload> struct SynthesisCandidate {
  CandidateClass candidateClass = CandidateClass::NativePassthrough;
  CandidateMetrics metrics;
  unsigned enumerationIndex = 0;
  Payload payload;
};

/// How to rewrite a single-qubit op onto the native menu.
///
/// - `Direct`: pattern-match the op type and emit the target gates directly
///   via `decomposeTo*` (applicable to a small fixed set of op types per
///   emitter).
/// - `MatrixFallback`: fold the op to a 2x2 matrix and run an Euler
///   decomposition in the emitter's basis; handles anything constant.
enum class SingleQubitRewriteStrategy : std::uint8_t { Direct, MatrixFallback };

/// Picked single-qubit rewrite: which emitter to use and how to drive it.
struct SingleQubitRewritePlan {
  SingleQubitRewriteStrategy strategy = SingleQubitRewriteStrategy::Direct;
  SingleQubitEmitterSpec emitter;
};

/// Picked two-qubit rewrite: a pre-computed abstract gate sequence produced
/// by `TwoQubitBasisDecomposer` plus the single-qubit emitter and entangler
/// basis used when materializing the sequence back into MLIR.
struct TwoQubitRewritePlan {
  std::shared_ptr<decomposition::TwoQubitGateSequence> sequence;
  SingleQubitEmitterSpec emitter;
  EntanglerBasis entanglerBasis = EntanglerBasis::None;
};

} // namespace mlir::qco::native_synth
