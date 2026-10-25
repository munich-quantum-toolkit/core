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

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLFunctionalExtras.h"

#include <cstdint>
#include <string_view>

namespace mlir {
class Value;

namespace qc {
class QCProgramBuilder;
} // namespace qc
} // namespace mlir

namespace mqt::bench::detail {

/// Prepare a big-endian register from '0', '1', and '+' input bits.
void prepareRegister(mlir::qc::QCProgramBuilder& builder, mlir::Value reg,
                     std::string_view bits);

/// Emit a loop whose phase angle follows a geometric sequence.
void phaseRotationLoop(
    mlir::qc::QCProgramBuilder& builder, mlir::Value lower, mlir::Value upper,
    mlir::Value step, mlir::Value initialAngle, mlir::Value scale,
    const mlir::function_ref<void(mlir::Value angle, mlir::Value index)>& body);

/// Emit addition or subtraction phases from low to high bits in [-2π, 2π].
void phaseAdditionLoop(
    mlir::qc::QCProgramBuilder& builder, int64_t width,
    const mlir::function_ref<mlir::Value(mlir::Value target)>& bit,
    const mlir::function_ref<void(mlir::Value angle, mlir::Value target)>& body,
    bool inverse = false);

/// Advance a phase for Fourier addition or semiclassical feedback.
///
/// With half = 0.5 and an initial correction of zero, firstAngle = ±π/2 bounds
/// feedback corrections by π; firstAngle = ±π bounds addition phases by 2π.
mlir::Value advancePhaseCorrection(mlir::qc::QCProgramBuilder& builder,
                                   mlir::Value correction, mlir::Value inputBit,
                                   mlir::Value half, mlir::Value firstAngle);

/// Apply the exact no-swap QFT or its inverse.
void qft(mlir::qc::QCProgramBuilder& builder, mlir::Value qubitRegister,
         int64_t qubits, bool inverse = false);

} // namespace mqt::bench::detail
