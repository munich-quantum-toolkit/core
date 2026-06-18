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
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>

/// Deterministic two-qubit lowering: Weyl decomposition + the
/// `TwoQubitBasisDecomposer` with a fixed entangler (CX before CZ) and the
/// first emitter's Euler basis for the surrounding single-qubit factors.

namespace mlir::qco::native_synth {

/// Number of entanglers (basis-gate uses) the minimal KAK decomposition of
/// `target` requires for the entangler selected by `spec` (CX before CZ).
/// Returns `std::nullopt` when `spec` has no usable entangler basis.
std::optional<std::uint8_t>
twoQubitEntanglerCount(const Matrix4x4& target, const NativeProfileSpec& spec);

/// Synthesize the two-qubit unitary `target` (raw `4×4`, any global phase) at
/// `(qubit0, qubit1)` into native entanglers and single-qubit gates of `spec`.
/// The entangler is chosen deterministically (CX before CZ) and the
/// single-qubit factors use the first emitter's Euler basis. Writes the output
/// qubit values to `outQubit0` / `outQubit1`.
///
/// Returns `failure()` when the profile has no usable entangler basis or the
/// KAK decomposition is not realizable with that entangler.
LogicalResult emitTwoQubitNative(IRRewriter& rewriter, Location loc,
                                 Value qubit0, Value qubit1,
                                 const Matrix4x4& target,
                                 const NativeProfileSpec& spec,
                                 Value& outQubit0, Value& outQubit1);

/// Rewrite `XXPlusYY` / `XXMinusYY` via two `RZZ` blocks (menus with `rzz`).
LogicalResult rewriteXXPlusMinusYYViaRzz(IRRewriter& rewriter, Operation* op);

} // namespace mlir::qco::native_synth
