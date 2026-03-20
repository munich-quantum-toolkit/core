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

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace mlir::qco {

/**
 * @brief Deterministic order for SSA values.
 *
 * @details Uses block order when both defining ops are in the same block;
 * otherwise fall back to opaque pointer order for a deterministic total order.
 */
struct SSAOrder {
  bool operator()(Value a, Value b) const {
    auto* opA = a.getDefiningOp();
    auto* opB = b.getDefiningOp();
    if (!opA || !opB || opA->getBlock() != opB->getBlock()) {
      return a.getAsOpaquePointer() < b.getAsOpaquePointer();
    }
    return opA->isBeforeInBlock(opB);
  }
};

} // namespace mlir::qco
