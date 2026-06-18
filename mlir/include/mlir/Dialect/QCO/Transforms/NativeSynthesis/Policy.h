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

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"

#include <mlir/IR/Operation.h>

/// Menu membership checks for native synthesis (no IR rewrites).

namespace mlir::qco::native_synth {

/// Whether the menu contains the corresponding two-qubit entangler. Used by
/// the 2q rewrite path to pick between CX and CZ emission.
bool usesCxEntangler(const NativeProfileSpec& spec);
bool usesCzEntangler(const NativeProfileSpec& spec);

/// Whether an already-lowered single-qubit op is in the menu (i.e. no
/// further rewrite needed).
bool allowsSingleQubitOp(UnitaryOpInterface op, const NativeProfileSpec& spec);

/// Whether `op` has a direct (non-matrix) lowering via the corresponding
/// `decomposeTo*` helper in `SingleQubit.h`. These are used for ops whose
/// angles are not compile-time constants, so no constant ``2×2`` matrix is
/// available for the matrix-driven path.
bool canDirectlyDecomposeToZSXX(Operation* op, bool supportsDirectRx);
bool canDirectlyDecomposeToU3(Operation* op);
bool canDirectlyDecomposeToR(Operation* op);
bool canDirectlyDecomposeToAxisPair(Operation* op, AxisPair axisPair);

} // namespace mlir::qco::native_synth
