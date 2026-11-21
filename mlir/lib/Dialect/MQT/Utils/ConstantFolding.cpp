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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
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

// Fold a detached operation: dialect folders may rewrite it in place.
static Attribute foldWithInputs(Operation* op, ValueRange inputs,
                                ArrayRef<Attribute> attributes) {
  OwningOpRef<Operation*> copy(Operation::create(
      op->getLoc(), op->getName(), op->getResultTypes(), inputs,
      op->getRawDictionaryAttrs(), op->getPropertiesStorage(), {}, 0));
  SmallVector<OpFoldResult, 1> results;
  if (!isSpeculatable(copy.get()) || failed(copy->fold(attributes, results)) ||
      results.size() != 1) {
    return {};
  }
  if (auto attr = dyn_cast_if_present<Attribute>(results.front())) {
    return attr;
  }
  // Identity folds return one of the detached operands.
  auto value = dyn_cast_if_present<Value>(results.front());
  const auto found = llvm::find(inputs, value);
  return found == inputs.end() ? Attribute{}
                               : attributes[found - inputs.begin()];
}

static Attribute
foldConstantOperation(Operation* op, Block& arguments,
                      const DenseMap<Value, Attribute>& cache) {
  SmallVector<Attribute> attributes;
  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    attributes.push_back(cache.lookup(operand));
    if (index == arguments.getNumArguments()) {
      arguments.addArgument(operand.getType(), op->getLoc());
    } else {
      arguments.getArgument(index).setType(operand.getType());
    }
  }
  // Most folders use only the operand attributes. Stand-in arguments keep the
  // source use lists untouched and avoid creating constant operations.
  if (auto attr = foldWithInputs(
          op, arguments.getArguments().take_front(attributes.size()),
          attributes)) {
    return attr;
  }

  // Some folders and speculation checks inspect the defining operations.
  Block constants;
  OpBuilder builder(op->getContext());
  builder.setInsertionPointToEnd(&constants);
  SmallVector<Value> inputs;
  for (auto [operand, attr] : llvm::zip_equal(op->getOperands(), attributes)) {
    auto* def = operand.getDefiningOp();
    Operation* literal = nullptr;
    if (def != nullptr && def->hasTrait<OpTrait::ConstantLike>()) {
      literal = builder.clone(*def);
    } else {
      if (def != nullptr && def->getDialect() != op->getDialect()) {
        literal = def->getDialect()->materializeConstant(
            builder, attr, operand.getType(), op->getLoc());
      }
      if (literal == nullptr) {
        literal = op->getDialect()->materializeConstant(
            builder, attr, operand.getType(), op->getLoc());
      }
    }
    if (literal == nullptr) {
      return {};
    }
    inputs.push_back(literal->getResult(0));
  }
  return foldWithInputs(op, inputs, attributes);
}

static Attribute evaluateConstant(Value value,
                                  DenseMap<Value, Attribute>* sharedCache) {
  if (isa<BlockArgument>(value)) {
    return sharedCache != nullptr ? sharedCache->lookup(value) : Attribute{};
  }
  Attribute literal;
  if (matchPattern(value, m_Constant(&literal))) {
    return literal;
  }
  // Start small for one-shot unary and binary expressions.
  DenseMap<Value, Attribute> localCache(sharedCache != nullptr ? 0 : 4);
  auto& cache = sharedCache != nullptr ? *sharedCache : localCache;
  if (const auto found = cache.find(value); found != cache.end()) {
    return found->second;
  }
  SmallVector<std::pair<Value, unsigned>> stack{{value, 0}};
  Block arguments;
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

    Attribute folded = foldConstantOperation(operation, arguments, cache);
    if (!folded) {
      return {};
    }
    cache[current] = folded;
    stack.pop_back();
  }
  return cache.lookup(value);
}

Attribute valueToConstantAttr(Value value, DenseMap<Value, Attribute>& cache) {
  return evaluateConstant(value, &cache);
}

Attribute valueToConstantAttr(Value value) {
  return evaluateConstant(value, nullptr);
}

std::optional<double> valueToConstantDouble(Value value) {
  return attributeToDouble(valueToConstantAttr(value));
}

} // namespace mlir::mqt
