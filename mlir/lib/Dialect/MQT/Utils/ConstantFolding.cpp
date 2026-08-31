/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/Utils/ConstantFolding.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <optional>

namespace mlir::mqt {

std::optional<double> attributeToDouble(Attribute attr) {
  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    return floatAttr.getValueAsDouble();
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
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

std::optional<Attribute>
valueToConstantAttr(Value value,
                    DenseMap<Value, std::optional<Attribute>>& cache) {
  if (const auto it = cache.find(value); it != cache.end()) {
    return it->second;
  }

  struct Frame {
    Value value;
    Operation* operation;
    unsigned nextOperand = 0;
  };

  SmallVector<Frame> worklist;
  llvm::SmallDenseSet<Value> active;
  const auto schedule = [&](Value candidate) {
    if (cache.contains(candidate)) {
      return;
    }
    Attribute attr;
    if (matchPattern(candidate, m_Constant(&attr))) {
      cache[candidate] = attr;
      return;
    }
    Operation* operation = candidate.getDefiningOp();
    if (operation == nullptr || operation->getNumRegions() != 0 ||
        !isPure(operation)) {
      cache[candidate] = std::nullopt;
      return;
    }
    active.insert(candidate);
    worklist.push_back({candidate, operation});
  };

  schedule(value);
  while (!worklist.empty()) {
    auto& frame = worklist.back();
    bool scheduledOperand = false;
    while (frame.nextOperand < frame.operation->getNumOperands()) {
      Value operand = frame.operation->getOperand(frame.nextOperand++);
      if (cache.contains(operand)) {
        continue;
      }
      if (active.contains(operand)) {
        cache[operand] = std::nullopt;
        continue;
      }
      schedule(operand);
      scheduledOperand = true;
      break;
    }
    if (scheduledOperand) {
      continue;
    }

    SmallVector<Attribute> operands;
    operands.reserve(frame.operation->getNumOperands());
    bool failedOperand = false;
    for (Value operand : frame.operation->getOperands()) {
      const auto it = cache.find(operand);
      if (it == cache.end() || !it->second) {
        failedOperand = true;
        break;
      }
      operands.push_back(*it->second);
    }
    if (failedOperand) {
      active.erase(frame.value);
      cache[frame.value] = std::nullopt;
      worklist.pop_back();
      continue;
    }

    SmallVector<OpFoldResult, 1> results;
    if (failed(frame.operation->fold(operands, results)) ||
        results.size() != 1) {
      active.erase(frame.value);
      cache[frame.value] = std::nullopt;
      worklist.pop_back();
      continue;
    }
    if (auto resultAttr = dyn_cast_if_present<Attribute>(results.front())) {
      active.erase(frame.value);
      cache[frame.value] = resultAttr;
      worklist.pop_back();
      continue;
    }

    auto resultValue = dyn_cast_if_present<Value>(results.front());
    if (!resultValue || resultValue == frame.value ||
        active.contains(resultValue)) {
      active.erase(frame.value);
      cache[frame.value] = std::nullopt;
      worklist.pop_back();
      continue;
    }
    if (!cache.contains(resultValue)) {
      schedule(resultValue);
      continue;
    }
    active.erase(frame.value);
    cache[frame.value] = cache.lookup(resultValue);
    worklist.pop_back();
  }

  return cache.lookup(value);
}

std::optional<Attribute> valueToConstantAttr(Value value) {
  DenseMap<Value, std::optional<Attribute>> cache;
  return valueToConstantAttr(value, cache);
}

std::optional<double> valueToConstantDouble(Value value) {
  if (const auto attr = valueToConstantAttr(value)) {
    return attributeToDouble(*attr);
  }
  return std::nullopt;
}

} // namespace mlir::mqt
