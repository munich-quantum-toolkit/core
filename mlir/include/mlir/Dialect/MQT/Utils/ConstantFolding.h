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

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Value.h>

#include <optional>

namespace mlir::mqt {

/// Convert a floating-point or integer attribute to a double.
[[nodiscard]] std::optional<double> attributeToDouble(Attribute attr);

/// Convert a direct arithmetic constant to a double.
[[nodiscard]] std::optional<double> valueToDouble(Value value);

/**
 * Recursively constant-fold a pure SSA expression DAG to an attribute.
 *
 * The cache memoizes successful and failed evaluations so shared operands are
 * resolved once.
 *
 * @param value SSA value to evaluate.
 * @param cache Evaluation results indexed by SSA value.
 */
[[nodiscard]] std::optional<Attribute>
valueToConstantAttr(Value value,
                    DenseMap<Value, std::optional<Attribute>>& cache);

/// Recursively constant-fold a pure SSA expression DAG to an attribute.
[[nodiscard]] std::optional<Attribute> valueToConstantAttr(Value value);

/// Recursively constant-fold a pure SSA expression DAG to a double.
[[nodiscard]] std::optional<double> valueToConstantDouble(Value value);

} // namespace mlir::mqt
