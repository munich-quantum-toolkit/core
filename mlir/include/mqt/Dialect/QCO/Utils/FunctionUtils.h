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

#include "llvm/ADT/SmallVector.h"

namespace mlir::qco {
/// Positions of scalar qubits and qubit tensors in argument order.
[[nodiscard]] SmallVector<unsigned> getQuantumArgumentIndices(TypeRange types);

/// Argument continued by a synthetic trailing result of an ordinary call.
[[nodiscard]] FailureOr<unsigned> getCallArgumentForResult(func::CallOp call,
                                                           unsigned result);

/// Synthetic trailing result of an ordinary call that continues an argument.
[[nodiscard]] FailureOr<unsigned> getCallResultForArgument(func::CallOp call,
                                                           unsigned argument);

/// Return the quantum argument continued by @p value.
///
/// QCO functions return one trailing value for every scalar qubit or register
/// argument, in argument order. Generic calls are followed only through that
/// ABI.
[[nodiscard]] FailureOr<unsigned> traceQubitArgument(func::FuncOp function,
                                                     Value value);

/// Return the value that @p value continues: a block argument, or the result
/// that created it, such as an allocation or an extracted register element.
///
/// The trace crosses a region operation only after proving that every region
/// hands the value back at its own position, so a branch or loop that
/// exchanges values fails the trace. Values the trace cannot follow, including
/// a cycle of values, fail rather than abort or loop, which makes this usable
/// on unverified IR.
[[nodiscard]] FailureOr<Value> traceQuantumOrigin(Value value);

/// Return the quantum block argument of @p block continued by @p value.
[[nodiscard]] FailureOr<unsigned> traceQubitArgument(Block& block, Value value);

/// Check that extracted tensor slots are restored at calls and region exits.
/// Matching slot indices are a program precondition; correspondence is checked
/// separately.
[[nodiscard]] bool hasCompleteTensorLifetime(Value tensor, unsigned depth = 0);
} // namespace mlir::qco
