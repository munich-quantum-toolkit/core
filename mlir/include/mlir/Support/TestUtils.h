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

namespace mlir {
class Block;
class ModuleOp;
class Operation;
class Region;

/**
 * @brief Compare two modules for structural equivalence using MLIR's APIs
 *
 * @details
 * Uses MLIR's IR walking and comparison mechanisms to verify that two
 * modules are structurally equivalent. This is more robust than string
 * comparison as it is insensitive to formatting differences.
 *
 * @param lhs First module to compare
 * @param rhs Second module to compare
 * @return true if modules are structurally equivalent, false otherwise
 */
bool modulesAreEquivalent(ModuleOp lhs, ModuleOp rhs);

/**
 * @brief Compare two operations for structural equivalence
 *
 * @details
 * Recursively compares operations, their attributes, operands, results,
 * and nested regions. This implements MLIR's structural equivalence check.
 *
 * @param lhs First operation to compare
 * @param rhs Second operation to compare
 * @return true if operations are structurally equivalent, false otherwise
 */
bool operationsAreEquivalent(Operation* lhs, Operation* rhs);

/**
 * @brief Compare two regions for structural equivalence
 *
 * @param lhs First region to compare
 * @param rhs Second region to compare
 * @return true if regions are structurally equivalent, false otherwise
 */
bool regionsAreEquivalent(Region* lhs, Region* rhs);

/**
 * @brief Compare two blocks for structural equivalence
 *
 * @param lhs First block to compare
 * @param rhs Second block to compare
 * @return true if blocks are structurally equivalent, false otherwise
 */
bool blocksAreEquivalent(Block* lhs, Block* rhs);

} // namespace mlir
