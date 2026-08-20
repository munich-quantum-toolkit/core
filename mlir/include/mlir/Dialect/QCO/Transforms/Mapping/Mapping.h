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

#include <mlir/Pass/Pass.h>

#include <memory>

namespace mlir {

class Block;
class CompilerTarget;

namespace qco {

namespace detail {

/**
 * Sort a block by SSA dependencies without changing classical-memory order.
 *
 * Return false and leave the block unchanged if sorting is impossible. This
 * includes a cycle in the combined dependency graph or more operations than a
 * 32-bit sort index can represent.
 */
[[nodiscard]] bool
sortTopologicallyPreservingClassicalMemoryOrder(Block* block);

} // namespace detail

/**
 * @brief Create a mapping pass instance for a compiler target.
 * @returns a pass object.
 */
std::unique_ptr<Pass> createMappingPass(const CompilerTarget& target,
                                        MappingPassOptions options);

} // namespace qco
} // namespace mlir
