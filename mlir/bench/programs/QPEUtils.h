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

namespace mlir {
class Value;

namespace qc {
class QCProgramBuilder;
} // namespace qc
} // namespace mlir

namespace mqt::bench::detail {

/// Estimate an eigenphase with a query register and the no-swap inverse QFT.
///
/// The callback applies the current controlled power and returns the value
/// representing the next doubled power.
void measureEigenphase(
    mlir::qc::QCProgramBuilder& builder, int64_t precision, mlir::Value result,
    mlir::Value initialPower,
    const mlir::function_ref<mlir::Value(mlir::Value control,
                                         mlir::Value power)>& applyPower);
} // namespace mqt::bench::detail
