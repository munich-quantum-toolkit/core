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

/// Single-qubit lowering: `decomposeTo*` for symbolic matches, plus
/// `computeSynthesizedSingleQubitLength` /
/// `emitSynthesizedSingleQubitFromMatrix` for the Euler matrix fallback.

namespace mlir::qco::native_synth {

/// Direct (non-matrix) single-qubit lowering to each single-qubit emission
/// strategy. Returns the output qubit value, or a null `Value` if no direct
/// rule applies and a matrix-based fallback must be tried.
///
/// When `supportsDirectRx` is true, `decomposeToZSXX` also passes `Rx`
/// through unchanged and lowers `Ry` / `R` via an `rz * rx * rz` sandwich.
Value decomposeToZSXX(IRRewriter& rewriter, Operation* op, Value inQubit,
                      bool supportsDirectRx);
Value decomposeToU3(IRRewriter& rewriter, Operation* op, Value inQubit);
Value decomposeToR(IRRewriter& rewriter, Operation* op, Value inQubit);
Value decomposeToAxisPair(IRRewriter& rewriter, Operation* op, Value inQubit,
                          AxisPair axisPair);

/// Cost estimate in number of emitted ops for fusing a single-qubit unitary
/// with the given emitter. Returns `SIZE_MAX` if no Euler basis is available.
std::size_t
computeSynthesizedSingleQubitLength(const Eigen::Matrix2cd& matrix,
                                    const SingleQubitEmitterSpec& emitter);

/// Emit the fused `2×2` unitary as native ops, inserting a `qco.gphase` if the
/// emitted sequence carries a non-trivial residual global phase.
Value emitSynthesizedSingleQubitFromMatrix(
    IRRewriter& rewriter, Location loc, Value inQubit,
    const Eigen::Matrix2cd& matrix, const SingleQubitEmitterSpec& emitter);

} // namespace mlir::qco::native_synth
