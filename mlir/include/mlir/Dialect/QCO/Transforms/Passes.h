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

#include <cstdint>
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

/// Create multi-controlled decomposition for one compiler target.
///
/// Supported operations remain native on all-to-all targets. Targets with
/// explicit connectivity use the target-independent decomposition.
[[nodiscard]] std::unique_ptr<Pass>
createDecomposeMultiControlled(const CompilerTarget& target,
                               uint64_t minQubits = 3);

/// Create post-routing synthesis for one immutable compiler target.
/// Each qubit must have a known static site. Structured branch exits must agree
/// on sites and loop backedges must preserve their entry sites.
/// The input may be modified on failure.
[[nodiscard]] std::unique_ptr<Pass>
createTargetNativeSynthesis(const CompilerTarget& target);

/// Create the final mapped-operation verifier, requiring known static sites.
[[nodiscard]] std::unique_ptr<Pass>
createVerifyTargetConformance(const CompilerTarget& target);

} // namespace mlir::qco
