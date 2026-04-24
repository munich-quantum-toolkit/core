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

#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include <optional>

/// Parses the pass `native-gates` string into a `NativeProfileSpec` (emitters,
/// entanglers, `allowedGates`). Token set matches `Passes.td` on this pass.

namespace mlir::qco::native_synth {

/// Euler bases that can reconstruct a two-axis single-qubit unitary.
llvm::SmallVector<decomposition::EulerBasis>
getEulerBasesForAxisPair(AxisPair axisPair);

/// Resolve a comma-separated native gate menu (e.g. `"x,sx,rz,cx"`) into a
/// full `NativeProfileSpec`.
///
/// Recognised tokens: `u`, `x`, `sx`, `rz` (or `p`), `rx`, `ry`, `r`,
/// `cx`, `cz`, `rzz`.
std::optional<NativeProfileSpec>
resolveNativeGatesSpec(llvm::StringRef nativeGates);

} // namespace mlir::qco::native_synth
