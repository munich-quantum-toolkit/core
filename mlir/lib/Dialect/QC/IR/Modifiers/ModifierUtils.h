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

#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace mlir {

class Block;
class Operation;
class Region;

namespace qc::detail {

/**
 * @brief Verify the operations and SSA captures in a QC modifier body.
 */
[[nodiscard]] LogicalResult verifyModifierBody(Operation* modifierOp,
                                               Block& body);

/**
 * @brief Report the region successors of a QC modifier operation.
 *
 * @details Entering @p modifierOp branches into @p body, forwarding the
 * operands aliased by its block arguments, and the body's terminator branches
 * back to @p modifierOp.
 */
void getModifierSuccessorRegions(Operation* modifierOp, Region& body,
                                 RegionBranchPoint point,
                                 SmallVectorImpl<RegionSuccessor>& regions);

} // namespace qc::detail

} // namespace mlir
