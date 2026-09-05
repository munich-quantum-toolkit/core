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

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

namespace mlir::qco {
/// Return the qubit argument continued by @p value.
///
/// QCO functions return one trailing qubit for every qubit argument, in
/// qubit-argument order. Generic calls are followed only through that ABI.
[[nodiscard]] FailureOr<unsigned> traceQubitArgument(func::FuncOp function,
                                                     Value value);
} // namespace mlir::qco
