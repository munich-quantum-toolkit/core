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
 * @brief Create target-independent two-qubit gate fusion.
 */
[[nodiscard]] std::unique_ptr<Pass> createFuseTwoQubitGates();

/**
 * @brief Create post-routing synthesis from the module target environment.
 */
[[nodiscard]] std::unique_ptr<Pass> createTargetNativeSynthesis();

/// Create target-native synthesis with an explicit target.
[[nodiscard]] std::unique_ptr<Pass>
createTargetNativeSynthesis(const CompilerTarget& target);

/**
 * @brief Create the final mapped-operation conformance verifier.
 */
[[nodiscard]] std::unique_ptr<Pass> createVerifyTargetConformance();

/// Create target conformance verification with an explicit target.
[[nodiscard]] std::unique_ptr<Pass>
createVerifyTargetConformance(const CompilerTarget& target);

} // namespace mlir::qco
