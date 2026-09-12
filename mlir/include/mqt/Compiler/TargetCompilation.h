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

#include "llvm/ADT/ArrayRef.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace mlir {

class TargetEnvironment;
class OpPassManager;

/// Controls for native placement and routing trials.
///
/// Set both fields for repeatable mapping across machines with different CPU
/// counts, using the same Core build, input, and target. All-to-all placement
/// ignores these controls. The selected layout may change between releases.
struct MappingOptions {
  size_t seed = 42;
  /// A positive count, or no value to use the available logical CPU count.
  std::optional<size_t> trials;
};

/// Snapshot of native placement and routing for the input allocation order.
///
/// Allocations follow entry-block order; tensor slots follow ascending index.
/// Each layout entry is a target site ID, not a dense hardware vertex. Idle
/// input qubits are included. The snapshot describes this compilation only;
/// later program transformations do not update it.
struct MappingResult {
  std::vector<size_t> allocationSizes;
  std::vector<int64_t> initialLayout;
  std::vector<int64_t> finalLayout;
};

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
                                       const TargetEnvironment& environment,
                                       const MappingOptions& mapping = {});

/// Populate target compilation while preserving and reporting input layout.
///
/// Input allocations must have fixed sizes and belong to the entry block.
/// An empty initial layout selects automatic placement; otherwise provide one
/// distinct target site ID per input qubit. The result is assigned only after
/// successful compilation and must outlive the pass manager. This pipeline
/// preserves idle input wires, so it can use more sites than ordinary
/// compilation.
void populateTargetCompilationWithLayoutPipeline(
    OpPassManager& pm, const TargetEnvironment& environment,
    MappingResult& result, llvm::ArrayRef<int64_t> initialLayout = {},
    const MappingOptions& mapping = {});

/// Populate target-native block synthesis without routing.
///
/// Requires an all-to-all target. Inlines calls, decomposes supported
/// multi-controlled gates, assigns static sites, resynthesizes constant
/// two-qubit runs in the native basis, and verifies target conformance. Input
/// must use structured QCO/SCF control flow. The supplied environment is
/// authoritative and must remain unchanged during pipeline execution.
void populateTargetSynthesisPipeline(OpPassManager& pm,
                                     const TargetEnvironment& environment);

} // namespace mlir
