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

#include "llvm/ADT/APInt.h"

#include <cstdint>

namespace mlir::qc {
class QCProgramBuilder;
} // namespace mlir::qc

namespace mqt::bench::detail {

/// Fourier phases for multiply-accumulate, followed by a row for the modulus.
void appendModularPhaseAngles(mlir::SmallVectorImpl<double>& angles,
                              llvm::APInt multiplier,
                              const llvm::APInt& modulus);

/// Add or subtract control * multiplier * multiplicand modulo N into
/// accumulator. The phase table starts at offset. The accumulator must be below
/// N and its overflow bit and the work qubit must start clean.
void multiplyAccumulate(mlir::qc::QCProgramBuilder& builder,
                        mlir::Value control, mlir::Value multiplicand,
                        mlir::Value accumulator, mlir::Value work,
                        mlir::Value angles, mlir::Value offset, int64_t bits,
                        bool inverse = false);

} // namespace mqt::bench::detail
