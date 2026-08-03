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

#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

#include <memory>

namespace mlir {
class CompilerTarget;
} // namespace mlir

namespace mlir::qco {

#define GEN_PASS_DECL
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc" // IWYU pragma: export

/**
 * @brief Create the target-independent pre-routing two-qubit optimizer.
 */
[[nodiscard]] std::unique_ptr<Pass> createOptimizeTwoQubitUnitaryRuns();

/**
 * @brief Create post-routing synthesis for one immutable compiler target.
 */
[[nodiscard]] std::unique_ptr<Pass>
createTargetNativeSynthesis(const CompilerTarget& target);

/**
 * @brief Create the final mapped-operation conformance verifier.
 */
[[nodiscard]] std::unique_ptr<Pass>
createVerifyTargetConformance(const CompilerTarget& target);

} // namespace mlir::qco
