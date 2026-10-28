/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>

namespace mlir::mqt {

std::optional<double> attributeToDouble(Attribute attr) {
  if (auto floatAttr = dyn_cast_if_present<FloatAttr>(attr)) {
    return floatAttr.getValueAsDouble();
  }
  if (auto intAttr = dyn_cast_if_present<IntegerAttr>(attr)) {
    const bool isSigned = !intAttr.getType().isUnsignedInteger();
    APFloat value(APFloat::IEEEdouble(), APInt::getZero(64));
    value.convertFromAPInt(intAttr.getValue(), isSigned,
                           APFloat::rmNearestTiesToEven);
    return value.convertToDouble();
  }
  return std::nullopt;
}

std::optional<double> valueToDouble(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr))) {
    return std::nullopt;
  }
  return attributeToDouble(attr);
}

Attribute valueToConstantAttr(Value value, DenseMap<Value, Attribute>& cache) {
  if (const auto found = cache.find(value); found != cache.end()) {
    return found->second;
  }
  SmallVector<std::pair<Value, unsigned>> stack{{value, 0}};
  SmallVector<Attribute> operands;
  SmallVector<OpFoldResult, 1> results;
  while (!stack.empty()) {
    auto [current, nextOperand] = stack.back();
    Operation* operation = current.getDefiningOp();
    if (nextOperand == 0) {
      // A failed dependency leaves every pending ancestor cached as failed.
      auto entry = cache.try_emplace(current).first;
      Attribute attr;
      if (matchPattern(current, m_Constant(&attr))) {
        entry->second = attr;
        stack.pop_back();
        continue;
      }
      if (operation == nullptr || operation->getNumRegions() != 0 ||
          !isPure(operation)) {
        return {};
      }
    }
    if (nextOperand < operation->getNumOperands()) {
      Value operand = operation->getOperand(stack.back().second++);
      const auto found = cache.find(operand);
      if (found == cache.end()) {
        stack.emplace_back(operand, 0);
      } else if (!found->second) {
        return {};
      }
      continue;
    }

    operands.clear();
    for (Value operand : operation->getOperands()) {
      operands.push_back(cache.lookup(operand));
    }
    results.clear();
    if (failed(operation->fold(operands, results)) || results.size() != 1) {
      return {};
    }
    if (auto attr = dyn_cast_if_present<Attribute>(results.front())) {
      cache[current] = attr;
    } else if (auto replacement = dyn_cast_if_present<Value>(results.front())) {
      // Identity folds can return an existing value outside the operands.
      const auto found = cache.find(replacement);
      if (found == cache.end()) {
        stack.emplace_back(replacement, 0);
        continue;
      }
      if (!found->second) {
        return {};
      }
      cache[current] = found->second;
    } else {
      return {};
    }
    stack.pop_back();
  }
  return cache.lookup(value);
}

Attribute valueToConstantAttr(Value value) {
  if (isa<BlockArgument>(value)) {
    return {};
  }
  Attribute literal;
  if (matchPattern(value, m_Constant(&literal))) {
    return literal;
  }
  // Start small for unary and binary expressions.
  DenseMap<Value, Attribute> cache(4);
  return valueToConstantAttr(value, cache);
}

std::optional<double> valueToConstantDouble(Value value) {
  return attributeToDouble(valueToConstantAttr(value));
}

} // namespace mlir::mqt
