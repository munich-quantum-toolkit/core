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
class RewriterBase;
namespace qco {
class CtrlOp;
} // namespace qco
} // namespace mlir

namespace mlir::mqt {

/// Distribute a verified composite control over its body; fail without changing
/// IR if the body has fewer than two unitaries.
[[nodiscard]] LogicalResult unrollControl(qco::CtrlOp op,
                                          RewriterBase& rewriter);

} // namespace mlir::mqt
