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

/**
 * @brief Options for the canonical compiler-target pipeline.
 */
struct TargetCompilationOptions {
  /// Preserve quantum operations whose final qubit values are unobserved.
  bool preserveUnobservedQuantumOperations = false;
};

/**
 * @brief Populate the canonical compiler-target pipeline.
 *
 * @details Decomposes supported multi-controlled gates, performs
 * target-independent optimization, maps to the target topology, synthesizes
 * native operations, performs a final local cleanup, and verifies target
 * conformance.
 */
void populateTargetCompilationPipeline(
    OpPassManager& pm, const CompilerTarget& target,
    const TargetCompilationOptions& options = {});

} // namespace mlir
