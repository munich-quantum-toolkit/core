/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// \file
/// Single-qubit native-synthesis lowering helpers.
/// Covers symbolic `decomposeTo*` rewrites (used for dynamic-angle ops) plus
/// the matrix-driven `emitSingleQubitMatrix` synthesizer that lowers any
/// constant ``2×2`` unitary via the shared `Euler.h` synthesis.

#pragma once

#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::qco::native_synth {

/// Direct (non-matrix) single-qubit lowering to the `ZSXX` emitter
/// (`{Rz, Sx, X}`). Returns the output qubit value, or a null `Value` if no
/// direct rule applies and a matrix-based fallback must be tried.
///
/// When `supportsDirectRx` is true, the emitter also passes `Rx` through
/// unchanged and lowers `Ry` / `R` via an `rz * rx * rz` sandwich.
Value decomposeToZSXX(IRRewriter& rewriter, Operation* op, Value inQubit,
                      bool supportsDirectRx);

/// Direct (non-matrix) single-qubit lowering to a `U(theta, phi, lambda)`
/// output. Returns the output qubit value, or a null `Value` if no direct
/// rule applies and a matrix-based fallback must be tried.
Value decomposeToU3(IRRewriter& rewriter, Operation* op, Value inQubit);

/// Direct (non-matrix) single-qubit lowering to the `R(theta, phi)` emitter.
/// Returns the output qubit value, or a null `Value` if no direct rule
/// applies and a matrix-based fallback must be tried.
Value decomposeToR(IRRewriter& rewriter, Operation* op, Value inQubit);

/// Direct (non-matrix) single-qubit lowering to a two-axis emitter
/// identified by `axisPair` (e.g. `{Rx, Rz}`, `{Ry, Rz}`). Returns the
/// output qubit value, or a null `Value` if no direct rule applies and a
/// matrix-based fallback must be tried.
Value decomposeToAxisPair(IRRewriter& rewriter, Operation* op, Value inQubit,
                          AxisPair axisPair);

/// Synthesize a constant ``2×2`` unitary `matrix` into native gates of `basis`
/// (including a `qco.gphase` when the residual phase is non-trivial) and
/// return the resulting output qubit. Wraps `decomposition::Euler`.
Value emitSingleQubitMatrix(IRRewriter& rewriter, Location loc, Value inQubit,
                            const Matrix2x2& matrix,
                            decomposition::EulerBasis basis);

} // namespace mlir::qco::native_synth
