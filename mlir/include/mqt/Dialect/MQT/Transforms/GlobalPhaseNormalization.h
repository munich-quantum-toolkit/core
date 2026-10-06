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

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class ModuleOp;
class RewriterBase;
} // namespace mlir

namespace mlir::mqt {

/// Normalize QC and QCO global phases in @p moduleOp.
[[nodiscard]] LogicalResult normalizeGlobalPhases(ModuleOp moduleOp);

/// Normalize QC and QCO global phases in @p moduleOp.
///
/// Use the caller's rewriter so its listener observes new wires and erasures.
[[nodiscard]] LogicalResult normalizeGlobalPhases(ModuleOp moduleOp,
                                                  RewriterBase& rewriter);

} // namespace mlir::mqt
