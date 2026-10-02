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

#include "mqt/Dialect/CBit/IR/CBitOps.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir {
class Operation;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace mlir::qc {

/// Find unique measurement destinations that can be stored immediately
/// after their measurement without crossing conflicting classical accesses.
/// Nonadjacent stores require static indices. Other measurement uses are
/// excluded unless the caller can preserve them after moving the store.
/// Apply every returned move in measurement order. The input IR is not
/// modified.
[[nodiscard]] llvm::DenseMap<Operation*, cbit::StoreOp>
findMeasurementStores(func::FuncOp function, bool allowOtherUses = false);

} // namespace mlir::qc
