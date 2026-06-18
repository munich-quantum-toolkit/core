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
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateSequence.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include <Eigen/Core>
#include <optional>

/// F64 helpers, global phase, SU(4) normalization, and 2q sequence emission.

namespace mlir::qco::native_synth {

/// Convert a compile-time QCO 2x2 matrix into Eigen form.
[[nodiscard]] Eigen::Matrix2cd toEigen(const Matrix2x2& matrix);

/// Convert a compile-time QCO 4x4 matrix into Eigen form.
[[nodiscard]] Eigen::Matrix4cd toEigen(const Matrix4x4& matrix);

/// Create an ``arith.constant`` F64.
Value createF64Const(IRRewriter& rewriter, Location loc, double value);

/// If ``value`` is an F64 ``arith.constant``, return its value.
std::optional<double> getConstantF64(Value value);

/// Emit a `qco.gphase` if `phase` is non-negligible.
void emitGPhaseIfNonTrivial(IRRewriter& rewriter, Location loc, double phase);

/// Matrix equality up to a unit-modulus global phase.
bool isEquivalentUpToGlobalPhase(const Eigen::Matrix4cd& lhs,
                                 const Eigen::Matrix4cd& rhs,
                                 double atol = 1e-10);

/// Rescale `matrix` to determinant 1 (SU(4)) for Weyl / basis decomposers.
/// No-op if det is numerically zero.
void normalizeToSU4(Eigen::Matrix4cd& matrix);

/// ``getUnitaryMatrix4x4`` then rescale to SU(4).
bool getNormalizedTwoQubitMatrix(UnitaryOpInterface unitary,
                                 Eigen::Matrix4cd& matrix);

/// 4x4 for a 2q block member (plain 2q, ``CtrlOp`` CX/CZ, or lifted 1q). Fails
/// for barriers, ``gphase``, multi-control, or non-constant matrix parameters.
bool getBlockTwoQubitMatrix(Operation* op, Eigen::Matrix4cd& matrix);

/// Emit `seq` in order.
LogicalResult
emitTwoQubitGateSequenceAtLoc(IRRewriter& rewriter, Location loc, Value qubit0,
                              Value qubit1,
                              const decomposition::TwoQubitGateSequence& seq,
                              Value& outQubit0, Value& outQubit1);

/// Emit a two-qubit gate sequence and replace `op` with the resulting tails.
LogicalResult
emitTwoQubitGateSequence(IRRewriter& rewriter, Operation* op, Value qubit0,
                         Value qubit1,
                         const decomposition::TwoQubitGateSequence& seq);

} // namespace mlir::qco::native_synth
