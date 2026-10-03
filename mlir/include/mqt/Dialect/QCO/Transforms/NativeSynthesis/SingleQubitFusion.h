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

#include "mqt/Compiler/Target.h"

#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>

namespace mlir {
class ModuleOp;
class Operation;
class RewritePatternSet;
class RewriterBase;
} // namespace mlir

namespace mlir::qco::decomposition {

/// Combine rotations with a common generator before native lowering.
/// With a target, only produce rotations whose angles are unrestricted.
void populateRotationCompositionPatterns(
    RewritePatternSet& patterns, const CompilerTarget* target = nullptr);

/// Fuse one wire at a time during an existing reverse-order traversal.
/// Reuse within one MLIR context.
/// The caller must visit users before producers: fusion can erase successors.
/// A supplied target must have a synthesis basis.
class SingleQubitRunFusion {
public:
  SingleQubitRunFusion(const CompilerTarget::SynthesisBasis& basis,
                       const CompilerTarget* target,
                       GreedyRewriteConfig config = {});

  /// Ignore non-heads; compose runtime runs before their constant segments.
  LogicalResult apply(Operation* operation);

private:
  CompilerTarget::SynthesisBasis basis_;
  const CompilerTarget* target_;
  GreedyRewriteConfig config_;
  std::optional<FrozenRewritePatternSet> runtimePatterns_;
  std::optional<FrozenRewritePatternSet> matrixPatterns_;
};

/// Populates @p patterns with the single-qubit run fusion rewrite for
/// @p basis (the reusable core of `fuse-single-qubit-unitary-runs`).
///
/// Requires a shorter run if every gate is native to @p target, or belongs to
/// @p basis when no target is supplied.
void populateFuseSingleQubitUnitaryRunsPatterns(
    RewritePatternSet& patterns, const CompilerTarget::SynthesisBasis& basis,
    const CompilerTarget* target = nullptr);

/// Populates patterns that compose profitable parameterized single-qubit runs.
///
/// The patterns emit @p basis directly. @p target supplies native gate support.
/// Native fusion preserves supported parameterized runs and uses direct Euler
/// identities. General quaternion composition is reserved for controlled U
/// bodies and standalone fusion without a target.
void populateParameterizedSingleQubitRunCompositionPatterns(
    RewritePatternSet& patterns, const CompilerTarget::SynthesisBasis& basis,
    const CompilerTarget* target = nullptr);

/// Synthesize an equatorial target, carrying Z frames through diagonal gates.
/// Runs only when the native single-qubit basis is R.
LogicalResult synthesizeEquatorialGates(RewriterBase& rewriter,
                                        ModuleOp moduleOp,
                                        const CompilerTarget& target,
                                        const GreedyRewriteConfig& config);

} // namespace mlir::qco::decomposition
