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

#include <mlir/Support/LogicalResult.h>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace mlir::mqt {

/// Normalize QC and QCO global phases in @p moduleOp.
[[nodiscard]] LogicalResult normalizeGlobalPhases(ModuleOp moduleOp);

} // namespace mlir::mqt
