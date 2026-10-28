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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::qco {
/// Collect quantum wire roots and verify positional function and region
/// results.
///
/// Requires ordinary MLIR verification first. Region arguments remain local
/// roots; verified region results continue their corresponding inputs.
[[nodiscard]] LogicalResult collectWireOrigins(Operation* root,
                                               DenseMap<Value, Value>& origins);

/// Positions of scalar qubits and qubit tensors in argument order.
[[nodiscard]] SmallVector<unsigned> getQuantumArgumentIndices(TypeRange types);

/// Argument continued by a synthetic trailing result of an ordinary call.
[[nodiscard]] FailureOr<unsigned> getCallArgumentForResult(func::CallOp call,
                                                           unsigned result);

/// Return the quantum argument continued by @p value.
///
/// QCO functions return one trailing value for every scalar qubit or register
/// argument, in argument order. Generic calls are followed only through that
/// ABI.
[[nodiscard]] FailureOr<unsigned> traceQubitArgument(func::FuncOp function,
                                                     Value value);

/// Return the quantum block argument continued by @p value.
[[nodiscard]] FailureOr<unsigned> traceQubitArgument(Block& block, Value value);

/// Check that extracted tensor slots are restored at calls and region exits.
///
/// Slot identity and disjoint extractions are program preconditions; known
/// violations and positional region correspondence are checked separately.
/// Returns false at nesting depth 64 or greater to bound recursive stack use.
[[nodiscard]] bool hasCompleteTensorLifetime(Value tensor, unsigned depth = 0);
} // namespace mlir::qco
