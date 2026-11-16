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

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::memref {
class AllocaOp;
} // namespace mlir::memref

namespace mlir::qc {

/// Resolve a fully initialized, immutable buffer of borrowed qubit references.
/// Each slot must have one constant-index store that dominates all loads.
[[nodiscard]] FailureOr<llvm::SmallVector<Value>>
collectQubitReferences(memref::AllocaOp storage);

} // namespace mlir::qc
