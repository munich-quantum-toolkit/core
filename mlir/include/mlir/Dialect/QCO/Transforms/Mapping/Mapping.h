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

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Region.h>

namespace mlir::qco {

// Forward declaration
class Architecture;

/**
 * @brief Verifies if all two-qubit gates within the region are executable on
 * the targeted architecture. Expects static qubits only.
 * @returns llvm::success() if executable, llvm::failure() otherwise.
 */
LogicalResult isExecutable(Region& region, const Architecture& arch);
} // namespace mlir::qco
