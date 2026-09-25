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
class InvOp;
class PowOp;
} // namespace qco
} // namespace mlir

namespace mlir::mqt {

/// Distribute a composite control in verified linear QCO IR; fail without
/// changing IR if the body has fewer than two unitaries.
[[nodiscard]] LogicalResult unrollModifier(qco::CtrlOp op,
                                           RewriterBase& rewriter);

/// Unroll an inverse body in reverse order. Requires verified linear QCO IR.
/// Fail without changing IR if it has fewer than two unitaries.
[[nodiscard]] LogicalResult unrollModifier(qco::InvOp op,
                                           RewriterBase& rewriter);

/// Distribute a power in verified linear QCO IR over disjoint body operations
/// for a constant integer exponent. Fail without changing IR for other or
/// noncomposite bodies.
[[nodiscard]] LogicalResult unrollModifier(qco::PowOp op,
                                           RewriterBase& rewriter);

} // namespace mlir::mqt
