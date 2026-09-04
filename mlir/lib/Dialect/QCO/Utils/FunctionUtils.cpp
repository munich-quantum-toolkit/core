/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Utils/FunctionUtils.h"

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/SymbolTable.h>

using namespace mlir;
using namespace mlir::qco;

FailureOr<unsigned> mlir::qco::traceQubitArgument(func::FuncOp function,
                                                  Value value) {
  if (function.isDeclaration()) {
    return failure();
  }
  while (true) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (argument.getOwner() == &function.getBody().front() &&
          isa<QubitType>(argument.getType())) {
        return argument.getArgNumber();
      }
      return failure();
    }

    if (auto call = value.getDefiningOp<func::CallOp>()) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee) {
        return failure();
      }
      SmallVector<unsigned> qubitArguments;
      for (auto [index, type] : llvm::enumerate(callee.getArgumentTypes())) {
        if (isa<QubitType>(type)) {
          qubitArguments.emplace_back(index);
        }
      }
      auto result = cast<OpResult>(value).getResultNumber();
      if (call.getNumResults() < qubitArguments.size() ||
          result < call.getNumResults() - qubitArguments.size()) {
        return failure();
      }
      value = call.getOperand(qubitArguments[result - (call.getNumResults() -
                                                       qubitArguments.size())]);
      continue;
    }

    WireIterator iterator(value);
    --iterator;
    if (iterator == std::default_sentinel) {
      return failure();
    }
    value = iterator.qubit();
  }
}
