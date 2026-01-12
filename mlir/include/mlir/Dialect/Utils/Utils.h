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

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Value.h>
#include <variant>

namespace mlir::utils {

constexpr auto TOLERANCE = 1e-15;

/**
 * @brief Convert a variant parameter (double or Value) to a Value
 *
 * @param builder The operation builder.
 * @param state The operation state.
 * @param parameter The parameter as a variant (double or Value).
 * @return Value The parameter as a Value.
 */
inline Value variantToValue(OpBuilder& builder, const OperationState& state,
                            const std::variant<double, Value>& parameter) {
  Value operand;
  if (std::holds_alternative<double>(parameter)) {
    operand = builder.create<arith::ConstantOp>(
        state.location, builder.getF64FloatAttr(std::get<double>(parameter)));
  } else {
    operand = std::get<Value>(parameter);
  }
  return operand;
}

inline Value constantFromScalar(OpBuilder& builder, Location loc, int64_t v) {
  return builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(v));
}

inline Value constantFromScalar(OpBuilder& builder, Location loc, bool v) {
  return builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(v));
}

/**
 * @brief Convert a variant parameter (T or Value) to a Value
 *
 * @param builder The operation builder.
 * @param state The location of the operation.
 * @param parameter The parameter as a variant (T or Value).
 * @return Value The parameter as a Value.
 */
template <typename T>
Value variantToValue(OpBuilder& builder, const Location loc,
                     const std::variant<T, Value>& parameter) {
  if (std::holds_alternative<Value>(parameter)) {
    return std::get<Value>(parameter);
  }
  return constantFromScalar(builder, loc, std::get<T>(parameter));
}
} // namespace mlir::utils
