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

/**
 * @brief Try to convert a float mlir::Value to a standard C++ double
 *
 * @details
 * Resolving the mlir::Value will only work if it is a static value, so a value
 * of type float and defined via a "arith.constant" operation.
 */
[[nodiscard]] inline std::optional<double> valueToDouble(Value value) {
  auto constantOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constantOp) {
    return std::nullopt;
  }
  auto&& floatAttr = dyn_cast<FloatAttr>(constantOp.getValue());
  if (!floatAttr) {
    return std::nullopt;
  }
  return floatAttr.getValueAsDouble();
}

} // namespace mlir::utils
