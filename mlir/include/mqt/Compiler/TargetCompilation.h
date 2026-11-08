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
#include "mqt/Compiler/TargetEnvironment.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

namespace mlir {

class OpPassManager;

/// Select the routing environment used by target compilation.
///
/// Restricts placement to usable sites and includes virtual resonator
/// couplings. Programs with no qubits retain the supplied environment.
[[nodiscard]] llvm::Expected<TargetEnvironment>
createTargetCompilationEnvironment(ModuleOp moduleOp,
                                   const TargetEnvironment& environment);

/// Register target preparation and finalization for textual pass pipelines.
///
/// Preparation uses the attached target environment, records source wire order,
/// and performs structural cleanup. Finalization synthesizes and verifies
/// native operations, then restores the execution environment saved by
/// preparation. Explicitly attaching another environment abandons that
/// restoration.
void registerTargetCompilationPasses();

/// Populate the canonical compiler-target pipeline.
///
/// Inlines reusable functions, decomposes supported multi-controlled gates,
/// maps device targets to their topology, fuses blocks in the target basis,
/// synthesizes native operations, performs a final local cleanup, and verifies
/// target conformance. The context that runs this low-level pipeline must
/// register inliner extensions for its callable dialects.
/// Input must use structured QCO/SCF control flow. Normalize CFG branches
/// before calling this pipeline. Runtime assertions are allowed.
/// Explicit native-operation targets must provide a single-qubit synthesis
/// basis so native synthesis can lower the results of standard cleanup.
/// The supplied environment is authoritative: the pipeline attaches it to the
/// module and shares its prepared target with every target-dependent pass.
/// The environment must remain unchanged during pipeline execution.
/// Use runWithCompilationOptions to apply compilation-wide seed and
/// instrumentation settings when running this pipeline.
/// Enable Pauli twirling to insert native single-qubit corrections around the
/// final eligible entanglers before target-conformance verification.
void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const TargetEnvironment& environment,
                                       const MappingOptions& mapping = {},
                                       bool enablePauliTwirling = false);

/// Populate target-native block synthesis without routing.
///
/// Device targets require all-to-all connectivity for dynamic qubits; static
/// qubits keep their site IDs and may use an explicit topology. Inlines calls,
/// decomposes
/// non-native controlled gates, places qubits for device targets, resynthesizes
/// constant two-qubit runs in the native basis, and verifies target
/// conformance. Input must use structured QCO/SCF control flow. The supplied
/// environment is authoritative and must remain unchanged during pipeline
/// execution. Explicit native-operation targets must provide a single-qubit
/// synthesis basis. Use runWithCompilationOptions to apply compilation-wide
/// seed and instrumentation settings when running this pipeline. Gate-set
/// targets preserve allocated program qubits and terminal permutations without
/// placement.
/// The Pauli-twirling option uses the same post-synthesis stage as compilation.
void populateTargetSynthesisPipeline(OpPassManager& pm,
                                     const TargetEnvironment& environment,
                                     const MappingOptions& mapping = {},
                                     bool enablePauliTwirling = false);

} // namespace mlir
