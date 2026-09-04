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

#include "mlir/Dialect/CBit/IR/CBitAttributes.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <cstdint>
#include <variant>

#define GET_OP_CLASSES
#include "mlir/Dialect/CBit/IR/CBitOps.h.inc" // IWYU pragma: export

namespace mlir::cbit {

/// Validates a static index used by a CBit register builder operation.
void validateStaticRegisterIndex(Value reg,
                                 const std::variant<int64_t, Value>& index);

/// Populates patterns that decompose whole-register operations into static
/// bit loads, stores, and ordinary integer arithmetic.
void populateCBitDecompositionPatterns(RewritePatternSet& patterns);

} // namespace mlir::cbit
