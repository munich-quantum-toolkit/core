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

#include <Eigen/Core>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>

#include <cstddef>
#include <optional>

/// Single-qubit lowering: `decomposeTo*` for symbolic matches, plus
/// `computeSynthesizedSingleQubitLength` /
/// `emitSynthesizedSingleQubitFromMatrix` for the Euler matrix fallback.

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

/// Euler sequence for matrix synthesis for non-`U3` emitters (same basis as
/// `emitSynthesizedSingleQubitFromMatrix`). `nullopt` for `U3` (single `u`
/// gate, no cached Euler list) or when the axis pair has no Euler basis.
std::optional<decomposition::QubitGateSequence>
eulerSequenceForMatrixSynthesis(const Eigen::Matrix2cd& matrix,
                                const SingleQubitEmitterSpec& emitter);

/// Cost estimate in number of emitted ops for fusing a single-qubit unitary
/// with the given emitter. Returns `SIZE_MAX` if no Euler basis is available.
std::size_t
computeSynthesizedSingleQubitLength(const Eigen::Matrix2cd& matrix,
                                    const SingleQubitEmitterSpec& emitter);

/// Emit the fused `2×2` unitary as native ops, inserting a `qco.gphase` if the
/// emitted sequence carries a non-trivial residual global phase.
Value emitSynthesizedSingleQubitFromMatrix(
    IRRewriter& rewriter, Location loc, Value inQubit,
    const Eigen::Matrix2cd& matrix, const SingleQubitEmitterSpec& emitter,
    const decomposition::QubitGateSequence* reuseEulerSeq = nullptr);

} // namespace mlir::qco::native_synth
