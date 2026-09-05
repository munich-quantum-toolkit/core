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

#include <cstdint>

namespace mlir {
class Value;

namespace qc {
class QCProgramBuilder;
} // namespace qc
} // namespace mlir

namespace mqt::bench::detail {

/// Apply the exact no-swap QFT used by the QFT-adder families.
void forwardQFT(mlir::qc::QCProgramBuilder& builder, mlir::Value qubitRegister,
                int64_t qubits);

/// Apply the exact inverse of `forwardQFT`.
void inverseQFT(mlir::qc::QCProgramBuilder& builder, mlir::Value qubitRegister,
                int64_t qubits);

} // namespace mqt::bench::detail
