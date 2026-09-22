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

#include <cstddef>
#include <cstdint>
#include <optional>
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

/// Apply the no-swap QFT, optionally omitting rotations beyond cutoff.
void forwardQFT(mlir::qc::QCProgramBuilder& builder, mlir::Value qubitRegister,
                int64_t qubits, std::optional<size_t> cutoff = std::nullopt);

/// Apply the adjoint of `forwardQFT` with the same cutoff.
void inverseQFT(mlir::qc::QCProgramBuilder& builder, mlir::Value qubitRegister,
                int64_t qubits, std::optional<size_t> cutoff = std::nullopt);

} // namespace mqt::bench::detail
