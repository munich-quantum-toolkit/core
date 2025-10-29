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

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <optional>
#include <string>

// Suppress warnings about ambiguous reversed operators in MLIR
// (see https://github.com/llvm/llvm-project/issues/45853)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#endif
#include <mlir/Interfaces/InferTypeOpInterface.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#define DIALECT_NAME_FLUX "flux"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Flux/IR/FluxOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Flux/IR/FluxOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

namespace mlir::flux {

template <typename ConcreteType>
class TargetArityTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TargetArityTrait> {};

template <typename ConcreteType>
class ParameterArityTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, ParameterArityTrait> {};

struct ParameterDescriptor {
  mlir::FloatAttr valueAttr;
  mlir::Value valueOperand;

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

} // namespace mlir::flux

#include "mlir/Dialect/Flux/IR/FluxInterfaces.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Flux/IR/FluxOps.h.inc" // IWYU pragma: export
