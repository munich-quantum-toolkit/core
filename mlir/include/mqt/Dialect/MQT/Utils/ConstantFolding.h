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

/// Constant-fold a pure SSA expression DAG, returning null on failure.
///
/// Nonconstant operations must be region-free and fold to exactly one result.
/// The cache stores a nonnull attribute on success and a null attribute on
/// failure. An absent entry has not been evaluated. Discard the cache when the
/// evaluated IR changes. Evaluation uses an explicit stack to support deep
/// expressions and stops at the first non-constant operand.
///
/// @param value SSA value to evaluate.
/// @param cache Evaluation results indexed by SSA value.
[[nodiscard]] Attribute valueToConstantAttr(Value value,
                                            DenseMap<Value, Attribute>& cache);

/// Constant-fold a pure SSA expression DAG, returning null on failure.
[[nodiscard]] Attribute valueToConstantAttr(Value value);

/// Constant-fold a pure SSA expression DAG to a double.
[[nodiscard]] std::optional<double> valueToConstantDouble(Value value);

} // namespace mlir::mqt
