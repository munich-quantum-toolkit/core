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

/// Add or subtract control * multiplier * multiplicand modulo N into
/// accumulator. The unsigned i64 multiplier and modulus are below 2^63.
/// The accumulator must be below N and its overflow bit and work qubit clean.
void multiplyAccumulate(mlir::qc::QCProgramBuilder& builder,
                        mlir::Value control, mlir::Value multiplicand,
                        mlir::Value accumulator, mlir::Value work,
                        mlir::Value multiplier, mlir::Value modulus,
                        int64_t bits, bool inverse = false);

} // namespace mqt::bench::detail
