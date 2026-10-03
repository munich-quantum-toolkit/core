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
} // namespace mlir

namespace mlir::qco::decomposition {

/// Run-rewrite choices resolved by the calling pass, independently of native
/// target capabilities. Individual lowering owns site-specific support.
struct SingleQubitFusionPolicy {
  enum class RuntimeExpressions { DirectOnly, ControlledBodies, General };

  bool preserveSingletons = false;
  bool skipControlledBodies = false;
  bool preserveNativeParameterizedRuns = false;
  RuntimeExpressions runtimeExpressions = RuntimeExpressions::General;

  /// Keep optional fusion exportable; general expressions remain available
  /// when a controlled body must be merged into a native U operation.
  static SingleQubitFusionPolicy
  forTarget(CompilerTarget::SingleQubitBasis basis) {
    const bool usesU = basis == CompilerTarget::SingleQubitBasis::U;
    return {
        .preserveSingletons = true,
        .skipControlledBodies = !usesU,
        .preserveNativeParameterizedRuns = true,
        .runtimeExpressions = usesU ? RuntimeExpressions::ControlledBodies
                                    : RuntimeExpressions::DirectOnly,
    };
  }
};

/// Fuse one wire at a time during an existing reverse-order traversal.
/// Reuse within one MLIR context.
/// The caller must visit users before producers: fusion can erase successors.
class SingleQubitRunFusion {
public:
  SingleQubitRunFusion(const CompilerTarget::SynthesisBasis& basis,
                       SingleQubitFusionPolicy policy,
                       const CompilerTarget* target,
                       GreedyRewriteConfig config = {});

  /// Ignore non-heads; compose runtime runs before their constant segments.
  LogicalResult apply(Operation* operation);

private:
  CompilerTarget::SynthesisBasis basis_;
  SingleQubitFusionPolicy policy_;
  const CompilerTarget* target_;
  GreedyRewriteConfig config_;
  std::optional<FrozenRewritePatternSet> runtimePatterns_;
  std::optional<FrozenRewritePatternSet> matrixPatterns_;
};

/// Standalone driver; target synthesis reuses its existing traversal instead.
LogicalResult fuseSingleQubitUnitaryRuns(
    ModuleOp moduleOp, const CompilerTarget::SynthesisBasis& basis,
    SingleQubitFusionPolicy policy, const CompilerTarget* target,
    const GreedyRewriteConfig& config);

/// Populates @p patterns with the single-qubit run fusion rewrite for
/// @p basis (the reusable core of `fuse-single-qubit-unitary-runs`).
///
/// Requires a shorter run if every gate is native to @p target, or belongs to
/// @p basis when no target is supplied.
void populateFuseSingleQubitUnitaryRunsPatterns(
    RewritePatternSet& patterns, const CompilerTarget::SynthesisBasis& basis,
    SingleQubitFusionPolicy policy = {},
    const CompilerTarget* target = nullptr);

/// Populates patterns that compose profitable parameterized single-qubit runs.
///
/// The patterns emit @p basis directly. @p target supplies native gate support;
/// @p policy selects controlled-body ownership, native-run preservation, and
/// whether fusion may emit general runtime expressions.
void populateParameterizedSingleQubitRunCompositionPatterns(
    RewritePatternSet& patterns, const CompilerTarget::SynthesisBasis& basis,
    SingleQubitFusionPolicy policy = {},
    const CompilerTarget* target = nullptr);

} // namespace mlir::qco::decomposition
