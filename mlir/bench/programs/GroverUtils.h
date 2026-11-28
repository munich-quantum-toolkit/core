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

#include "llvm/ADT/ArrayRef.h"

#include <string_view>

namespace mlir::qc {
class QCProgramBuilder;
} // namespace mlir::qc

namespace mqt::bench::detail {

void maskMarkedState(mlir::qc::QCProgramBuilder& builder,
                     llvm::ArrayRef<mlir::Value> search,
                     std::string_view markedBitstring);

/// Negate the marked state, optionally conditioned on a phase-estimation qubit.
void markPhase(mlir::qc::QCProgramBuilder& builder,
               llvm::ArrayRef<mlir::Value> search,
               std::string_view markedBitstring, mlir::Value control = {});

} // namespace mqt::bench::detail
