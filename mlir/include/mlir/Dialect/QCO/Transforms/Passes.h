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
}

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

} // namespace mlir::qco
