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

#include <llvm/ADT/StringRef.h>

namespace mlir::qc::openqasm {

constexpr llvm::StringLiteral SCALAR_ATTR = "mqt.openqasm.scalar";
constexpr llvm::StringLiteral ANGLE_VALUE_ATTR = "mqt.openqasm.angle";
constexpr llvm::StringLiteral ANGLE_OPERANDS_ATTR =
    "mqt.openqasm.angle_operands";
constexpr llvm::StringLiteral BIT_EXTRACT_ATTR = "mqt.openqasm.bit_extract";
constexpr llvm::StringLiteral UINT_VALUE_ATTR = "mqt.openqasm.uint";

} // namespace mlir::qc::openqasm
