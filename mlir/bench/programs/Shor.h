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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include <cstdint>

namespace mlir::qc {
class QCProgramBuilder;
} // namespace mlir::qc

namespace mqt::bench::detail {

/// Build Shor's controlled in-place multiplier from consecutive phase-table
/// rows for a multiplier and its modular inverse.
mlir::func::FuncOp createInPlaceMultiplier(mlir::qc::QCProgramBuilder& builder,
                                           int64_t bits,
                                           mlir::RankedTensorType anglesType);

} // namespace mqt::bench::detail
