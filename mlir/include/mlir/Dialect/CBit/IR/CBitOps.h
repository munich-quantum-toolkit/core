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
#include <mlir/Dialect/Arith/IR/Arith.h>
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

/// Maps signed ordering to the corresponding unsigned predicate.
arith::CmpIPredicate getUnsignedPredicate(arith::CmpIPredicate predicate);

/// Whether a value is a fixed-width bit vector rooted in a register read.
bool isRegisterBitVector(Value value);

/// Populates patterns that decompose whole-register operations into static
/// bit loads, stores, and ordinary integer arithmetic.
void populateCBitDecompositionPatterns(RewritePatternSet& patterns);

/// Builds an equivalent comparison from individual register bits.
Value buildComparison(OpBuilder& builder, Location location,
                      arith::CmpIPredicate predicate, const llvm::APInt& rhs,
                      llvm::function_ref<Value(int64_t)> loadBit);

} // namespace mlir::cbit
