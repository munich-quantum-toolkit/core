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

/// Find measurement destinations that can be stored immediately after
/// their measurement without crossing conflicting classical accesses.
///
/// Additional uses may reuse the destination bit when it is not overwritten
/// and derived values remain within the block. With `allowOtherUses`, skip this
/// check; the caller must preserve all other uses. Nonadjacent stores require
/// static indices. Apply every returned move in measurement order. The input IR
/// is not modified.
[[nodiscard]] llvm::DenseMap<Operation*, cbit::StoreOp>
findMeasurementStores(func::FuncOp function, bool allowOtherUses = false);

} // namespace mlir::qc
