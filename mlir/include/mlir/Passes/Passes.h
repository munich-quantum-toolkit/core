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

#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace mlir {

class RewritePatternSet;

} // namespace mlir

namespace mlir::qco {

#define GEN_PASS_DECL
#include "mlir/Passes/Passes.h.inc" // IWYU pragma: export

void populateGateDecompositionPatterns(
    mlir::RewritePatternSet& patterns,
    llvm::Statistic& twoQubitCreationTime,
    llvm::Statistic& numberOfTwoQubitCreations,
    llvm::Statistic& successfulSingleQubitDecompositions,
    llvm::Statistic& totalSingleQubitDecompositions,
    llvm::Statistic& successfulTwoQubitDecompositions,
    llvm::Statistic& totalTwoQubitDecompositions,
    llvm::Statistic& totalCircuitCollections,
    llvm::Statistic& totalTouchedGates,
    llvm::Statistic& subCircuitComplexityChange,
    llvm::Statistic& timeInCircuitCollection,
    llvm::Statistic& timeInSingleQubitDecomposition,
    llvm::Statistic& timeInTwoQubitDecomposition);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Passes/Passes.h.inc" // IWYU pragma: export

} // namespace mlir::qco
