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
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
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

static Value constantOperand(OpBuilder& builder, Operation* op, Value operand,
                             Attribute attr) {
  auto* def = operand.getDefiningOp();
  if (def != nullptr && def->hasTrait<OpTrait::ConstantLike>()) {
    return builder.clone(*def)->getResult(0);
  }
  auto materialize = [&](Dialect* dialect) {
    return dialect->materializeConstant(builder, attr, operand.getType(),
                                        op->getLoc());
  };
  auto* producer = def != nullptr ? def->getDialect() : nullptr;
  if (producer != nullptr && producer != op->getDialect()) {
    if (auto* literal = materialize(producer)) {
      return literal->getResult(0);
    }
  }
  if (auto* literal = materialize(op->getDialect())) {
    return literal->getResult(0);
  }
  // A folder can use the attribute even if neither dialect has a literal.
  return builder.getInsertionBlock()->addArgument(operand.getType(),
                                                  op->getLoc());
}

static Attribute foldWithAttributes(Operation* op,
                                    ArrayRef<Attribute> attributes) {
  SmallVector<OpFoldResult, 1> results;
  if (!isSpeculatable(op) || failed(op->fold(attributes, results)) ||
      results.size() != 1) {
    return {};
  }
  if (auto attr = dyn_cast<Attribute>(results.front())) {
    return attr;
  }
  Value replacement = cast<Value>(results.front());
  for (auto [operand, attr] : llvm::zip_equal(op->getOperands(), attributes)) {
    if (operand == replacement) {
      return attr;
    }
  }
  return {};
}

static Attribute foldConstantOperation(Operation* op,
                                       DenseMap<Value, Attribute>& cache,
                                       bool allowInPlace) {
  if (allowInPlace) {
    const auto attributes =
        llvm::map_to_vector(op->getOperands(), [&](Value operand) {
          return cache.lookup(operand);
        });
    if (auto attr = foldWithAttributes(op, attributes)) {
      return attr;
    }
  }

  Block scratch;
  OpBuilder builder(op->getContext());
  builder.setInsertionPointToEnd(&scratch);
  IRMapping mapping;
  SmallVector<Attribute> attributes;
  for (Value operand : op->getOperands()) {
    // An in-place fold can introduce operands outside the cached expression.
    auto attr = valueToConstantAttr(operand, cache);
    if (!attr) {
      return {};
    }
    attributes.push_back(attr);
    if (!mapping.contains(operand)) {
      mapping.map(operand, constantOperand(builder, op, operand, attr));
    }
  }

  // A dialect may fold in place. The clone has no uses of source values.
  Operation* copy = builder.clone(*op, mapping);
  return foldWithAttributes(copy, attributes);
}

static Attribute evaluate(Value value, DenseMap<Value, Attribute>& cache,
                          bool allowInPlace) {
  if (const auto found = cache.find(value); found != cache.end()) {
    return found->second;
  }
  SmallVector<std::pair<Value, unsigned>> stack{{value, 0}};
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
          operation->getNumResults() != 1 ||
          operation->getNumSuccessors() != 0 ||
          !isMemoryEffectFree(operation)) {
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

    Attribute folded = foldConstantOperation(operation, cache, allowInPlace);
    if (!folded) {
      return {};
    }
    cache[current] = folded;
    stack.pop_back();
  }
  return cache.lookup(value);
}

Attribute valueToConstantAttr(Value value, DenseMap<Value, Attribute>& cache) {
  return evaluate(value, cache, /*allowInPlace=*/false);
}

Attribute foldToConstantAttr(Value value, DenseMap<Value, Attribute>& cache) {
  return evaluate(value, cache, /*allowInPlace=*/true);
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
