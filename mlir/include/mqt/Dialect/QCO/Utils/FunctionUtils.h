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

/// Return the quantum argument continued by @p value.
///
/// QCO functions return one trailing value for every scalar qubit or register
/// argument, in argument order. Generic calls are followed only through that
/// ABI.
[[nodiscard]] FailureOr<unsigned> traceQubitArgument(func::FuncOp function,
                                                     Value value);

/// Check that extracted tensor slots are restored at calls and region exits.
/// Positional region and function-result correspondence is checked separately.
[[nodiscard]] bool hasCompleteTensorLifetime(Value tensor, unsigned depth = 0);
} // namespace mlir::qco
