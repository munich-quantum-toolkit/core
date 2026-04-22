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

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"

#include <mlir/IR/Operation.h>

#include <optional>

/// Menu checks and cost hints for synthesis candidates (no IR rewrites).

namespace mlir::qco::native_synth {

/// Score weights are valid iff they are finite and non-negative.
bool areValidScoreWeights(const ScoreWeights& weights);

/// Whether the menu contains the corresponding two-qubit entangler. Used by
/// the 2q rewrite path to pick between CX and CZ emission.
bool usesCxEntangler(const NativeProfileSpec& spec);
bool usesCzEntangler(const NativeProfileSpec& spec);

/// Whether an already-lowered single-qubit op is in the menu (i.e. no
/// further rewrite needed). `BarrierOp` / `GPhaseOp` always pass through
/// unchanged.
bool allowsSingleQubitOp(UnitaryOpInterface op, const NativeProfileSpec& spec);

/// Count 1q/2q gates and compute the depth of a gate sequence.
CandidateMetrics
computeGateSequenceMetrics(const decomposition::QubitGateSequence& seq);

/// Whether `op` has a direct (non-matrix) lowering via the corresponding
/// `decomposeTo*` helper in `SingleQubit.h`.
bool canDirectlyDecomposeToZSXX(Operation* op, bool supportsDirectRx);
bool canDirectlyDecomposeToU3(Operation* op);
bool canDirectlyDecomposeToR(Operation* op);
bool canDirectlyDecomposeToAxisPair(Operation* op, AxisPair axisPair);

/// Estimated metrics for the direct and matrix-fallback lowerings.
CandidateMetrics
estimateDirectSingleQubitMetrics(Operation* op,
                                 const SingleQubitEmitterSpec& emitter);
std::optional<CandidateMetrics>
estimateMatrixSingleQubitMetrics(UnitaryOpInterface unitary,
                                 const SingleQubitEmitterSpec& emitter);

} // namespace mlir::qco::native_synth
