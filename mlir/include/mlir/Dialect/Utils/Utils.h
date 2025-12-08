/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

namespace mlir::utils {

constexpr double TOLERANCE = 1e-12;

/**
 * @brief Convert a variant parameter (double or Value) to a Value
 *
 * @param builder The operation builder.
 * @param state The operation state.
 * @param parameter The parameter as a variant (double or Value).
 * @return Value The parameter as a Value.
 */
inline Value variantToValue(OpBuilder& builder, OperationState& state,
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

} // namespace mlir::utils
