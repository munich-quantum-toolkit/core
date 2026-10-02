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

#include "llvm/ADT/ArrayRef.h"

#include <string_view>

namespace mlir {
class Value;

namespace qc {
class QCProgramBuilder;
} // namespace qc
} // namespace mlir

namespace mqt::bench::detail {

void maskMarkedState(mlir::qc::QCProgramBuilder& builder,
                     llvm::ArrayRef<mlir::Value> search,
                     std::string_view markedBitstring);

void markPhase(mlir::qc::QCProgramBuilder& builder,
               llvm::ArrayRef<mlir::Value> search,
               std::string_view markedBitstring);

} // namespace mqt::bench::detail
