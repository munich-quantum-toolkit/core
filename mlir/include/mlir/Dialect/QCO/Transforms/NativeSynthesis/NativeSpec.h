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

#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"

#include <llvm/ADT/StringRef.h>

#include <optional>

namespace mlir::qco::native_synth {

/// Euler basis used to synthesize an arbitrary single-qubit unitary into the
/// gates emitted by `emitter`. This is the deterministic replacement for the
/// scored multi-basis search.
[[nodiscard]] decomposition::EulerBasis
emitterEulerBasis(const SingleQubitEmitterSpec& emitter);

/// Resolve a comma-separated native gate menu (e.g. `"x,sx,rz,cx"`) into a
/// full `NativeProfileSpec`.
///
/// Parses the pass `native-gates` string into a `NativeProfileSpec`
/// (single-qubit emitters, entangler bases, and `allowedGates`). Token set
/// matches `Passes.td` on this pass.
///
/// Recognised tokens: `u`, `x`, `sx`, `rz` (or `p`), `rx`, `ry`, `r`,
/// `cx`, `cz`, `rzz`.
std::optional<NativeProfileSpec>
resolveNativeGatesSpec(llvm::StringRef nativeGates);

} // namespace mlir::qco::native_synth
