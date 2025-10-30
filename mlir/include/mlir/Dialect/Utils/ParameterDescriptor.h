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

class ParameterDescriptor {
  mlir::FloatAttr valueAttr;
  mlir::Value valueOperand;

public:
  ParameterDescriptor(mlir::FloatAttr attr = nullptr,
                      mlir::Value operand = nullptr) {
    assert(!(attr && operand) && "Cannot have both static and dynamic values");
    valueAttr = attr;
    valueOperand = operand;
  }

  bool isStatic() const { return valueAttr != nullptr; }
  bool isDynamic() const { return valueOperand != nullptr; }

  double getValueDouble() const {
    if (isDynamic()) {
      llvm_unreachable("Cannot get double value from dynamic parameter");
    }
    return valueAttr.getValueAsDouble();
  }
  mlir::FloatAttr getValueAttr() const { return valueAttr; }
  mlir::Value getValueOperand() const { return valueOperand; }
};

} // namespace mlir::utils
