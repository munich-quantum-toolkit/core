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

class CompilerTarget;
class OpPassManager;

/// Populate the canonical compiler-target pipeline.
///
/// Inlines reusable functions, decomposes supported multi-controlled gates,
/// performs target-independent optimization, maps to the target topology,
/// synthesizes native operations, performs a final local cleanup, and verifies
/// target conformance. The context that runs this low-level pipeline must
/// register inliner extensions for its callable dialects.
/// The input module must have an attached `mqt.target_env` whose compiler
/// target matches `target`. Use `attachTargetEnvironment` before running the
/// pipeline.
void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const CompilerTarget& target);

} // namespace mlir
