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

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>

namespace mlir::qc {
class QCProgramBuilder;
} // namespace mlir::qc

namespace mqt::bench::detail {

/// Add or subtract control * multiplier * multiplicand modulo modulus into
/// accumulator.
///
/// Requires 1 <= bits <= 63 and unsigned i64 residues 0 < multiplier < modulus
/// < 2^bits. The accumulator must be below the modulus, with its overflow bit
/// and work qubit at zero.
void multiplyAccumulate(mlir::qc::QCProgramBuilder& builder,
                        mlir::Value control, mlir::Value multiplicand,
                        mlir::Value accumulator, mlir::Value work,
                        mlir::Value multiplier, mlir::Value modulus,
                        int64_t bits, bool inverse = false);

} // namespace mqt::bench::detail
