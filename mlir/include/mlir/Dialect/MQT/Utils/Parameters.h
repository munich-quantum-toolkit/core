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

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <variant>

namespace mlir::mqt {

/// Absolute tolerance used when comparing static operation parameters.
inline constexpr double PARAMETER_COMPARISON_TOLERANCE = 1e-15;

/// Materialize a scalar as an arithmetic constant.
[[nodiscard]] Value constantFromScalar(OpBuilder& builder, Location loc,
                                       double value);

/// Materialize a scalar as an arithmetic constant.
[[nodiscard]] Value constantFromScalar(OpBuilder& builder, Location loc,
                                       int64_t value);

/// Materialize a scalar as an arithmetic constant.
[[nodiscard]] Value constantFromScalar(OpBuilder& builder, Location loc,
                                       bool value);

/// Convert a scalar or existing SSA value to an SSA value.
template <typename T>
[[nodiscard]] Value variantToValue(OpBuilder& builder, Location loc,
                                   const std::variant<T, Value>& parameter) {
  if (const auto* value = std::get_if<Value>(&parameter)) {
    return *value;
  }
  return constantFromScalar(builder, loc, std::get<T>(parameter));
}

/// Verify that each statically known floating-point parameter is finite.
[[nodiscard]] LogicalResult
verifyFiniteConstantParameters(Operation* operation, ValueRange parameters);

} // namespace mlir::mqt
