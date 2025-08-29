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

#include <mlir/Pass/Pass.h>

namespace mlir {

class RewritePatternSet;

} // namespace mlir

namespace mqt::ir::opt {

#define GEN_PASS_DECL
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc" // IWYU pragma: export

void populateGateEliminationPatterns(mlir::RewritePatternSet& patterns);
void populateMergeRotationGatesPatterns(mlir::RewritePatternSet& patterns);
void populateElidePermutationsPatterns(mlir::RewritePatternSet& patterns);
void populateSwapReconstructionPatterns(mlir::RewritePatternSet& patterns);
void populateAdvancedSwapReconstructionPatterns(
    mlir::RewritePatternSet& patterns);
void populateQuantumSinkShiftPatterns(mlir::RewritePatternSet& patterns);
void populateQuantumSinkPushPatterns(mlir::RewritePatternSet& patterns);
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
