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

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir::qco {
/// Verify linear uses, static roots, and positional quantum state flow.
///
/// Requires ordinary MLIR verification first. Every linear QCO value has
/// exactly one use, static indices are unique in the entry block, and function
/// and structured-control-flow results continue quantum arguments in order.
/// In SSA regions, equal constant indices or the same dynamic SSA index cannot
/// extract an outstanding register slot. Other dynamic aliasing remains a
/// program precondition.
[[nodiscard]] LogicalResult verifyLinearity(Operation* root);

/// Collect quantum wire roots and verify positional results and known
/// register-slot ownership in SSA regions.
///
/// Requires ordinary MLIR verification first. Region arguments remain local
/// roots; verified region results continue their corresponding inputs.
[[nodiscard]] LogicalResult collectWireOrigins(Operation* root,
                                               DenseMap<Value, Value>& origins);

/// Check that extracted tensor slots are restored at calls and region exits.
///
/// Slot identity and disjoint extractions are program preconditions; known
/// violations and positional region correspondence are checked separately.
/// Returns false at nesting depth 64 or greater to bound recursive stack use.
[[nodiscard]] bool hasCompleteTensorLifetime(Value tensor, unsigned depth = 0);
} // namespace mlir::qco
