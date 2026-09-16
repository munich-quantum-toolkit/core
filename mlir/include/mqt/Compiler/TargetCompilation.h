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

#include "mqt/Compiler/CompilationOptions.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir {

class TargetEnvironment;
class OpPassManager;

/// Initial and final target site IDs, including idle inputs.
///
/// Entries follow entry-block allocation order and ascending tensor slots.
/// Later transformations do not update this snapshot.
struct MappingResult {
  std::vector<size_t> allocationSizes;
  std::vector<int64_t> initialLayout;
  std::vector<int64_t> finalLayout;
};

/// Populate the canonical compiler-target pipeline.
///
/// Inlines reusable functions, decomposes supported multi-controlled gates,
/// maps to the target topology, fuses blocks in the target basis,
/// synthesizes native operations, performs a final local cleanup, and verifies
/// target conformance. The context that runs this low-level pipeline must
/// register inliner extensions for its callable dialects.
/// Input must use structured QCO/SCF control flow. Normalize CFG branches
/// before calling this pipeline. Runtime assertions are allowed.
/// The supplied environment is authoritative: the pipeline attaches it to the
/// module and shares its prepared target with every target-dependent pass.
/// The environment must remain unchanged during pipeline execution.
/// Use runWithCompilationOptions to apply compilation-wide seed and
/// instrumentation settings when running this pipeline.
void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const TargetEnvironment& environment,
                                       const MappingOptions& mapping = {});

/// Populate target compilation with input layout tracking.
///
/// Requires fixed-size local entry-block allocations. An empty initial layout
/// selects automatic placement; otherwise supply one distinct target site ID
/// per input. Idle slots count against capacity without adding operations.
/// The result must outlive the pass manager and is written only on success.
/// Run with runWithCompilationOptions to apply the shared seed and
/// instrumentation.
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
/// Use runWithCompilationOptions to apply compilation-wide seed and
/// instrumentation settings when running this pipeline.
void populateTargetSynthesisPipeline(OpPassManager& pm,
                                     const TargetEnvironment& environment,
                                     const MappingOptions& mapping = {});

} // namespace mlir
