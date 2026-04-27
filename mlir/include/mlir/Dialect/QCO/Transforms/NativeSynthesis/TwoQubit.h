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

#include <Eigen/Core>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <optional>

/// Two-qubit lowering: Weyl decomposition + `TwoQubitBasisDecomposer` over
/// each `(entangler, emitter Euler basis, basis-use count 0..3)` allowed by
/// the menu; the scorer picks the cheapest exact sequence.

namespace mlir::qco::native_synth {

/// Whether every gate in `seq` is allowed by `spec`'s menu.
bool gateSequenceFitsMenu(const decomposition::TwoQubitGateSequence& seq,
                          const NativeProfileSpec& spec);

/// Decompose a `4×4` target unitary into a gate sequence targeting the given
/// entangler basis, using `TwoQubitWeylDecomposition` +
/// `TwoQubitBasisDecomposer` with the supplied Euler basis.
std::optional<decomposition::TwoQubitGateSequence>
decomposeTwoQubitFromMatrix(const Eigen::Matrix4cd& matrix,
                            EntanglerBasis entangler,
                            decomposition::EulerBasis eulerBasis,
                            std::optional<std::uint8_t> numBasisUses);

/// Enumerate all direct + matrix-fallback single-qubit rewrite candidates.
llvm::SmallVector<SynthesisCandidate<SingleQubitRewritePlan>>
collectSingleQubitCandidates(UnitaryOpInterface unitary,
                             const NativeProfileSpec& spec);

/// Enumerate full two-qubit basis-decomposer candidates for a given `4×4`
/// target.
llvm::SmallVector<SynthesisCandidate<TwoQubitRewritePlan>, 0>
collectTwoQubitBasisCandidatesFromMatrix(const Eigen::Matrix4cd& targetMatrix,
                                         const NativeProfileSpec& spec);

/// Overload that reads the target matrix from a two-qubit op.
llvm::SmallVector<SynthesisCandidate<TwoQubitRewritePlan>, 0>
collectTwoQubitBasisCandidates(UnitaryOpInterface unitary,
                               const NativeProfileSpec& spec);

/// Scoring metrics for the `rewriteXXPlusMinusYYViaRzz` lowering (both
/// `XXPlusYY` and `XXMinusYY` branches emit the same gate counts).
CandidateMetrics xxPlusMinusYyRzzRewriteScoringMetrics();

/// Rewrite `XXPlusYY` / `XXMinusYY` via two `RZZ` blocks (menus with `rzz`).
LogicalResult rewriteXXPlusMinusYYViaRzz(IRRewriter& rewriter, Operation* op);

} // namespace mlir::qco::native_synth
