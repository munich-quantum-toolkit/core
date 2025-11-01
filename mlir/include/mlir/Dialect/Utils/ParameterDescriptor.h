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

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>

namespace mlir::utils {

/**
 * @brief Descriptor for a parameter that can be either static or dynamic
 */
class ParameterDescriptor {
  mlir::FloatAttr valueAttr;
  mlir::Value valueOperand;

public:
  /**
   * @brief Construct a new ParameterDescriptor object
   *
   * @param attr Static float attribute (optional)
   * @param operand Dynamic value operand (optional)
   */
  ParameterDescriptor(mlir::FloatAttr attr = nullptr,
                      mlir::Value operand = nullptr) {
    assert(!(attr && operand) && "Cannot have both static and dynamic values");
    valueAttr = attr;
    valueOperand = operand;
  }

  /**
   * @brief Check if the parameter is static
   *
   * @return true if static, false if dynamic
   */
  bool isStatic() const { return valueAttr != nullptr; }

  /**
   * @brief Check if the parameter is dynamic
   *
   * @return true if dynamic, false if static
   */
  bool isDynamic() const { return valueOperand != nullptr; }

  /**
   * @brief Try to get the double value of the parameter
   *
   * @return double value
   */
  double getValueDouble() const {
    if (isDynamic()) {
      llvm_unreachable("Cannot get double value from dynamic parameter");
    }
    return valueAttr.getValueAsDouble();
  }

  /**
   * @brief Get the static float attribute
   *
   * @return mlir::FloatAttr Static float attribute (nullptr if dynamic)
   */
  mlir::FloatAttr getValueAttr() const { return valueAttr; }

  /**
   * @brief Get the dynamic value operand
   *
   * @return mlir::Value Dynamic value operand (nullptr if static)
   */
  mlir::Value getValueOperand() const { return valueOperand; }
};

} // namespace mlir::utils
