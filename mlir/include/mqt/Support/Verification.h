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
class Operation;

namespace mqt {

/// Check finite constant expressions used by QC and QCO gate parameters.
/// Verify MLIR operation structure first. Call before transforming or exporting
/// a program. Walk pure, region-free expression graphs, including all select
/// operands. Dynamic values must be finite at runtime. Cache entries exist only
/// during this call; no IR mutation may run concurrently with this check.
[[nodiscard]] LogicalResult verifyProgramParameters(Operation* root);

} // namespace mqt
} // namespace mlir
