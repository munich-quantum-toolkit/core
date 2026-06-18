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
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <mlir/IR/PatternMatch.h>

#include <optional>

/// F64 helpers and block unitary extraction for native gate synthesis.

namespace mlir::qco::native_synth {

/// Create an ``arith.constant`` F64.
Value createF64Const(IRRewriter& rewriter, Location loc, double value);

/// If ``value`` is an F64 ``arith.constant``, return its value.
std::optional<double> getConstantF64(Value value);

/// Emit a `qco.gphase` if `phase` is non-negligible.
void emitGPhaseIfNonTrivial(IRRewriter& rewriter, Location loc, double phase);

/// 4x4 for a 2q block member (plain 2q, ``CtrlOp`` CX/CZ, or lifted 1q). Fails
/// for barriers, ``gphase``, multi-control, or non-constant matrix parameters.
bool getBlockTwoQubitMatrix(Operation* op, Matrix4x4& matrix);

} // namespace mlir::qco::native_synth
