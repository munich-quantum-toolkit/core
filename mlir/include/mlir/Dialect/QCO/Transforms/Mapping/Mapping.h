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

#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Region.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace mlir::qco {

// Forward declaration
class Architecture;

/**
 * @brief Verifies if all two-qubit gates within the region are executable on
 * the targeted architecture. Expects static qubits only.
 * @returns llvm::success() if executable, llvm::failure() otherwise.
 */
LogicalResult isExecutable(Region& region, const Architecture& arch);

/**
 * @brief Create a mapping pass instance with the given target architecture.
 * @returns a pass object.
 */
std::unique_ptr<Pass>
createMappingPass(std::shared_ptr<Architecture> arch,
                  MappingPassOptions options = MappingPassOptions{});
} // namespace mlir::qco
