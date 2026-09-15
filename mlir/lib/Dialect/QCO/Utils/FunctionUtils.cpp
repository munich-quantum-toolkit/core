/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Utils/FunctionUtils.h"

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Utils/WireIterator.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"
#include "mqt/Dialect/QTensor/Utils/TensorIterator.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"

#include <iterator>

using namespace mlir;
using namespace mlir::qco;

SmallVector<unsigned> mlir::qco::getQuantumArgumentIndices(TypeRange types) {
  SmallVector<unsigned> arguments;
  for (auto [index, type] : llvm::enumerate(types)) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (isa<QubitType>(type) ||
        (tensor && isa<QubitType>(tensor.getElementType()))) {
      arguments.emplace_back(index);
    }
  }
  return arguments;
}

FailureOr<unsigned> mlir::qco::getCallArgumentForResult(func::CallOp call,
                                                        unsigned result) {
  if (!SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr())) {
    return failure();
  }
  auto arguments = getQuantumArgumentIndices(call.getOperandTypes());
  if (call.getNumResults() < arguments.size()) {
    return failure();
  }
  const auto first = call.getNumResults() - arguments.size();
  if (result < first || result >= call.getNumResults()) {
    return failure();
  }
  const auto argument = arguments[result - first];
  if (call.getResult(result).getType() != call.getOperand(argument).getType()) {
    return failure();
  }
  return argument;
}

static FailureOr<unsigned> traceBlockArgument(Block& block, Value value) {
  while (true) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (argument.getOwner() == &block &&
          !getQuantumArgumentIndices(TypeRange{argument.getType()}).empty()) {
        return argument.getArgNumber();
      }
      return failure();
    }

    if (auto call = value.getDefiningOp<func::CallOp>()) {
      auto argument = getCallArgumentForResult(
          call, cast<OpResult>(value).getResultNumber());
      if (failed(argument)) {
        return failure();
      }
      value = call.getOperand(*argument);
      continue;
    }

    if (auto loop = value.getDefiningOp<scf::WhileOp>()) {
      auto result = cast<OpResult>(value).getResultNumber();
      auto argument = traceBlockArgument(
          *loop.getBeforeBody(), loop.getConditionOp().getArgs()[result]);
      if (failed(argument)) {
        return failure();
      }
      value = loop.getInits()[*argument];
      continue;
    }

    if (auto tensor = dyn_cast<TypedValue<RankedTensorType>>(value)) {
      qtensor::TensorIterator iterator(tensor);
      --iterator;
      if (iterator == std::default_sentinel || iterator.tensor() == value) {
        return failure();
      }
      value = iterator.tensor();
    } else {
      WireIterator iterator(value);
      --iterator;
      if (iterator == std::default_sentinel) {
        return failure();
      }
      value = iterator.qubit();
    }
  }
}

FailureOr<unsigned> mlir::qco::traceQubitArgument(func::FuncOp function,
                                                  Value value) {
  if (function.isDeclaration()) {
    return failure();
  }
  return traceBlockArgument(function.getBody().front(), value);
}

/// Require dynamic tensor slots to be restored before leaving each region.
/// Positional region correspondence is checked separately.
bool mlir::qco::hasCompleteTensorLifetime(Value tensor, unsigned depth) {
  /// ponytail: reject deeper nesting; use a worklist if proving
  /// completeness beyond 64 nested regions becomes necessary.
  if (depth == 64) {
    return false;
  }
  const auto isTensor = [](Value value) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    return type && isa<qco::QubitType>(type.getElementType());
  };
  DenseSet<llvm::PointerUnion<Attribute, Value>> extracted;
  while (tensor.hasOneUse()) {
    auto* user = *tensor.user_begin();
    if (auto extract = dyn_cast<qtensor::ExtractOp>(user)) {
      if (!extracted.insert(getAsOpFoldResult(extract.getIndex())).second) {
        return false;
      }
      tensor = extract.getOutTensor();
    } else if (auto insert = dyn_cast<qtensor::InsertOp>(user)) {
      if (!extracted.erase(getAsOpFoldResult(insert.getIndex()))) {
        return false;
      }
      tensor = insert.getResult();
    } else if (isa<scf::ForOp, scf::WhileOp, qco::IfOp, qco::IndexSwitchOp>(
                   user)) {
      const auto index =
          llvm::count_if(user->getOperands().take_front(
                             tensor.use_begin()->getOperandNumber()),
                         isTensor);
      for (Region& region : user->getRegions()) {
        auto arguments =
            llvm::filter_to_vector(region.getArguments(), isTensor);
        if (index >= arguments.size() ||
            !hasCompleteTensorLifetime(arguments[index], depth + 1)) {
          return false;
        }
      }
      auto results = llvm::filter_to_vector(user->getResults(), isTensor);
      if (index >= results.size()) {
        return false;
      }
      tensor = results[index];
    } else if (auto call = dyn_cast<func::CallOp>(user)) {
      if (!extracted.empty()) {
        return false;
      }
      auto arguments = qco::getQuantumArgumentIndices(call.getOperandTypes());
      auto* position =
          llvm::find(arguments, tensor.use_begin()->getOperandNumber());
      if (position == arguments.end() ||
          call.getNumResults() < arguments.size()) {
        return false;
      }
      const auto result = call.getNumResults() - arguments.size() +
                          std::distance(arguments.begin(), position);
      auto argument = qco::getCallArgumentForResult(call, result);
      if (failed(argument)) {
        return false;
      }
      tensor = call.getResult(result);
    } else {
      return isa<qtensor::DeallocOp, qco::YieldOp, scf::YieldOp,
                 scf::ConditionOp, func::ReturnOp>(user) &&
             extracted.empty();
    }
  }
  return false;
}
