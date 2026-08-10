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

#include <mlir/Support/LogicalResult.h>

namespace mlir {

class Block;
class Operation;

namespace qco::detail {

/**
 * @brief Verify the operations and SSA captures in a QCO modifier body.
 */
[[nodiscard]] LogicalResult verifyModifierBody(Operation* modifierOp,
                                               Block& body);

} // namespace qco::detail

} // namespace mlir
