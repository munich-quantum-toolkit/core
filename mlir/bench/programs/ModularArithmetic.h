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

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/APInt.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir::qc {
class QCProgramBuilder;
} // namespace mlir::qc
namespace mlir::func {
class FuncOp;
} // namespace mlir::func

namespace mqt::bench::detail {

/// Fourier phases for multiply-accumulate, followed by a row for the modulus.
void appendModularPhaseAngles(mlir::SmallVectorImpl<double>& angles,
                              llvm::APInt multiplier,
                              const llvm::APInt& modulus,
                              std::optional<size_t> cutoff = std::nullopt);

/// Add or subtract control * multiplier * multiplicand modulo N into
/// accumulator. The phase table starts at offset. The accumulator must be below
/// N and its overflow bit and the work qubit must start clean.
void multiplyAccumulate(mlir::qc::QCProgramBuilder& builder,
                        mlir::Value control, mlir::Value multiplicand,
                        mlir::Value accumulator, mlir::Value work,
                        mlir::Value angles, mlir::Value offset, int64_t bits,
                        bool inverse = false,
                        std::optional<size_t> cutoff = std::nullopt);

/// Create reusable private helpers for controlled in-place modular
/// multiplication. Arguments are control, n-bit value, n+1-bit zero
/// accumulator, zero work qubit, phase table, and offset. Tables for a and its
/// modular inverse are consecutive. Requires a coprime to N and value < N.
/// Exact arithmetic restores workspace to zero.
mlir::func::FuncOp
createInPlaceMultiplier(mlir::qc::QCProgramBuilder& builder, int64_t bits,
                        mlir::RankedTensorType anglesType,
                        std::optional<size_t> cutoff = std::nullopt);

} // namespace mqt::bench::detail
