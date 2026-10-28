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

#include "mqt/Compiler/ResultLayout.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/LogicalResult.h"

namespace mlir {

/// Replace terminal measurements and classical stores with dense raw readout.
///
/// Requires static qubits and straight-line terminal binary outputs. Physical
/// order sorts slots by site ID; otherwise source output order is preferred.
[[nodiscard]] FailureOr<ResultLayout>
normalizeTerminalReadout(ModuleOp moduleOp, bool physicalOrder,
                         bool measureAll = false);

} // namespace mlir
