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

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <optional>

namespace mlir::mqt {

/// Convert a floating-point or integer attribute to a double.
[[nodiscard]] inline std::optional<double> attributeToDouble(Attribute attr) {
  if (const auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    return floatAttr.getValueAsDouble();
  }
  if (const auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    const bool isSigned = !intAttr.getType().isUnsignedInteger();
    APFloat value(APFloat::IEEEdouble(), APInt::getZero(64));
    value.convertFromAPInt(intAttr.getValue(), isSigned,
                           APFloat::rmNearestTiesToEven);
    return value.convertToDouble();
  }
  return std::nullopt;
}

/// Convert a direct arithmetic constant to a double.
[[nodiscard]] inline std::optional<double> valueToDouble(Value value) {
  auto constantOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constantOp) {
    return std::nullopt;
  }
  return attributeToDouble(constantOp.getValue());
}

/// Recursively constant-fold a pure SSA expression DAG to an attribute.
///
/// @p cache memoizes successful and failed evaluations so shared operands are
/// resolved once.
[[nodiscard]] inline std::optional<Attribute>
valueToConstantAttr(Value value,
                    DenseMap<Value, std::optional<Attribute>>& cache) {
  if (const auto it = cache.find(value); it != cache.end()) {
    return it->second;
  }

  Attribute attr;
  if (matchPattern(value, m_Constant(&attr))) {
    return cache[value] = attr;
  }

  Operation* operation = value.getDefiningOp();
  if (operation == nullptr || operation->getNumRegions() != 0 ||
      !isPure(operation)) {
    return cache[value] = std::nullopt;
  }

  SmallVector<Attribute> operands;
  operands.reserve(operation->getNumOperands());
  for (const Value operand : operation->getOperands()) {
    const auto folded = valueToConstantAttr(operand, cache);
    if (!folded) {
      return cache[value] = std::nullopt;
    }
    operands.push_back(*folded);
  }

  SmallVector<OpFoldResult, 1> results;
  if (failed(operation->fold(operands, results)) || results.size() != 1) {
    return cache[value] = std::nullopt;
  }
  std::optional<Attribute> folded;
  if (const auto resultAttr = dyn_cast_if_present<Attribute>(results.front())) {
    folded = resultAttr;
  } else if (const auto resultValue =
                 dyn_cast_if_present<Value>(results.front())) {
    /// Identity-style folds can return an existing SSA value.
    folded = valueToConstantAttr(resultValue, cache);
  }
  return cache[value] = folded;
}

/// Recursively constant-fold a pure SSA expression DAG to an attribute.
[[nodiscard]] inline std::optional<Attribute> valueToConstantAttr(Value value) {
  DenseMap<Value, std::optional<Attribute>> cache;
  return valueToConstantAttr(value, cache);
}

/// Recursively constant-fold a pure SSA expression DAG to a double.
[[nodiscard]] inline std::optional<double> valueToConstantDouble(Value value) {
  if (const auto attr = valueToConstantAttr(value)) {
    return attributeToDouble(*attr);
  }
  return std::nullopt;
}

} // namespace mlir::mqt
