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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"

#include <optional>

namespace mlir::mqt {

/// Convert a floating-point or integer attribute to a double, if present.
[[nodiscard]] std::optional<double> attributeToDouble(Attribute attr);

/// Convert a direct constant-like value to a double.
[[nodiscard]] std::optional<double> valueToDouble(Value value);

/// Evaluate a constant SSA expression without changing the IR.
///
/// Requires single-result, region-free operations that are memory-effect-free
/// and speculatable with their constant operands. Supports deep expressions and
/// returns null at the first unknown operand or unsuccessful fold.
/// The cache holds an attribute on success and a null attribute on failure.
/// Discard the cache when the evaluated IR changes.
///
/// @param value SSA value to evaluate.
/// @param cache Evaluation results indexed by SSA value.
[[nodiscard]] Attribute valueToConstantAttr(Value value,
                                            DenseMap<Value, Attribute>& cache);

/// Evaluate a constant expression with exclusive access to its IR. Dialect
/// folds may rewrite defining operations in place. Discard the cache after
/// other changes to the IR.
[[nodiscard]] Attribute foldToConstantAttr(Value value,
                                           DenseMap<Value, Attribute>& cache);

/// Evaluate a constant SSA expression without changing the IR, returning null
/// on failure.
[[nodiscard]] Attribute valueToConstantAttr(Value value);

/// Return the numeric value of a constant SSA expression, if available.
[[nodiscard]] std::optional<double> valueToConstantDouble(Value value);

} // namespace mlir::mqt
