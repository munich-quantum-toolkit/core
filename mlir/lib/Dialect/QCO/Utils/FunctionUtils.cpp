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
#include "mqt/Dialect/QCO/Utils/WireIterator.h"
#include "mqt/Dialect/QTensor/Utils/TensorIterator.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

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

FailureOr<unsigned> mlir::qco::traceQubitArgument(Block& block, Value value) {
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
      auto argument = traceQubitArgument(
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
  return traceQubitArgument(function.getBody().front(), value);
}
