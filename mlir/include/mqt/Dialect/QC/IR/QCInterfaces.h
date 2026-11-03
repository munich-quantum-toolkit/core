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

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include <cstddef>

namespace mlir::qc {

/// Reject identical qubit references or static indices within one operation.
///
/// Other alias relationships remain runtime preconditions.
LogicalResult verifyDistinctQubitOperands(Operation* op, ValueRange operands);

LogicalResult verifyUnitaryOpInterface(Operation* op);

} // namespace mlir::qc

// clang-format:off
#include "mqt/Dialect/QC/IR/QCInterfaces.h.inc" // IWYU pragma: export
// clang-format:on
