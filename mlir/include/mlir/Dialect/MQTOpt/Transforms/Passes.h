/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "ir/QuantumComputation.hpp"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"

#include <cstdint>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

namespace mlir {

class RewritePatternSet;

} // namespace mlir
/**
 * Populate patterns that merge adjacent rotation gates into equivalent
 * rotations.
 * @param patterns Pattern set to populate with rotation-merge patterns.
 */
/**
 * Populate patterns that reconstruct and elide swaps when possible.
 * @param patterns Pattern set to populate with swap reconstruction and elision
 * patterns.
 */
/**
 * Populate patterns that shift quantum sink operations to enable optimizations.
 * @param patterns Pattern set to populate with quantum sink shift patterns.
 */
/**
 * Populate patterns that push quantum sink operations toward sinks for
 * simplification.
 * @param patterns Pattern set to populate with quantum sink push patterns.
 */
/**
 * Populate patterns that lift measurements above control operations when safe.
 * @param patterns Pattern set to populate with measurement-lifting patterns.
 */
/**
 * Populate patterns that replace controls on basis-state values with
 * conditional `if` constructs.
 * @param patterns Pattern set to populate with replacement patterns for
 * basis-state controls.
 */
/**
 * Populate patterns that lift measurements above generic quantum gates when
 * valid.
 * @param patterns Pattern set to populate with measurement-vs-gate reordering
 * patterns.
 */
/**
 * Populate patterns that eliminate dead (unused) gates from circuits.
 * @param patterns Pattern set to populate with dead-gate-elimination patterns.
 */
/**
 * Populate patterns that identify and enable reuse of physical qubits within a
 * circuit.
 * @param patterns Pattern set to populate with qubit-reuse patterns.
 */
/**
 * Populate patterns that convert MLIR MQTOpt operations into a
 * QuantumComputation representation.
 * @param patterns Pattern set to populate with conversion-to-QuantumComputation
 * patterns.
 * @param circuit Target QuantumComputation to populate or update with converted
 * operations.
 */
/**
 * Populate patterns that convert a QuantumComputation into MLIR MQTOpt
 * operations.
 * @param patterns Pattern set to populate with
 * conversion-from-QuantumComputation patterns.
 * @param circuit Source QuantumComputation whose contents are converted into
 * MLIR ops.
 */

namespace mqt::ir::opt {

/**
 * Placement strategies for mapping logical qubits to physical locations.
 *
 * Random: assign qubits using a random permutation.
 * Identity: keep logical qubit indices unchanged.
 */
enum class PlacementStrategy : std::uint8_t { Random, Identity };

#define GEN_PASS_DECL
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc" // IWYU pragma: export

void populateGateEliminationPatterns(mlir::RewritePatternSet& patterns);
void populateMergeRotationGatesPatterns(mlir::RewritePatternSet& patterns);
void populateSwapReconstructionAndElisionPatterns(
    mlir::RewritePatternSet& patterns);
void populateQuantumSinkShiftPatterns(mlir::RewritePatternSet& patterns);
void populateQuantumSinkPushPatterns(mlir::RewritePatternSet& patterns);
void populateLiftMeasurementsAboveControlsPatterns(
    mlir::RewritePatternSet& patterns);
void populateReplaceBasisStateControlsWithIfPatterns(
    mlir::RewritePatternSet& patterns);
void populateLiftMeasurementsAboveGatesPatterns(
    mlir::RewritePatternSet& patterns);
void populateDeadGateEliminationPatterns(mlir::RewritePatternSet& patterns);
void populateReuseQubitsPatterns(mlir::RewritePatternSet& patterns);
void populateToQuantumComputationPatterns(mlir::RewritePatternSet& patterns,
                                          qc::QuantumComputation& circuit);
void populateFromQuantumComputationPatterns(mlir::RewritePatternSet& patterns,
                                            qc::QuantumComputation& circuit);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc" // IWYU pragma: export
} // namespace mqt::ir::opt
