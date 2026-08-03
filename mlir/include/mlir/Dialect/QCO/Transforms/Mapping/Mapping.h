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

#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Region.h>
#include <mlir/Pass/Pass.h>

#include <cstddef>
#include <memory>
#include <utility>

namespace mlir {

class CompilerTarget;

namespace qco {

/**
 * @brief Create a mapping pass instance for a compiler target.
 * @returns a pass object.
 */
std::unique_ptr<Pass> createMappingPass(const CompilerTarget& target,
                                        MappingPassOptions options);

/**
 * @brief Create a mapping pass instance for a legacy symmetric coupling set.
 * @returns a pass object.
 */
std::unique_ptr<Pass>
createMappingPass(const llvm::DenseSet<std::pair<size_t, size_t>>& couplingSet,
                  MappingPassOptions options);

} // namespace qco
} // namespace mlir
