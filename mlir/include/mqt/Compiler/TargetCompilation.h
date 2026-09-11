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

namespace mlir {

class TargetEnvironment;
class OpPassManager;

/// Populate the canonical compiler-target pipeline.
///
/// Inlines reusable functions, decomposes supported multi-controlled gates,
/// performs target-independent optimization, maps to the target topology,
/// synthesizes native operations, performs a final local cleanup, and verifies
/// target conformance. The context that runs this low-level pipeline must
/// register inliner extensions for its callable dialects.
/// Input must use structured QCO/SCF control flow. Normalize CFG branches
/// before calling this pipeline. Runtime assertions are allowed.
/// The supplied environment is authoritative: the pipeline attaches it to the
/// module and shares its prepared target with every target-dependent pass.
/// The environment must remain unchanged during pipeline execution.
void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const TargetEnvironment& environment);

/// Populate target-native synthesis without the optimization or routing stages.
///
/// Requires an all-to-all target. Inlines calls, decomposes supported
/// multi-controlled gates, assigns static sites, synthesizes native operations,
/// and verifies target conformance. Input must use structured QCO/SCF control
/// flow. The supplied environment is authoritative and must remain unchanged
/// during pipeline execution.
void populateTargetSynthesisPipeline(OpPassManager& pm,
                                     const TargetEnvironment& environment);

} // namespace mlir
