/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QTensor/Utils/CallTensorMapping.h"

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QTensor/Utils/TensorIterator.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

namespace mlir::qtensor {

// Returns whether a type is a tensor of qubits.
static bool isQubitTensor(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && isa<qco::QubitType>(tensorType.getElementType());
}

// Returns the position of a value among the qubit tensors in a range.
static std::optional<size_t> tensorPositionIn(ValueRange range, Value value) {
  size_t position = 0;
  for (Value candidate : range) {
    if (!isQubitTensor(candidate.getType())) {
      continue;
    }
    if (candidate == value) {
      return position;
    }
    ++position;
  }
  return std::nullopt;
}

FailureOr<int64_t> CallTensorMapping::threadToResult(Value argument,
                                                     func::ReturnOp returnOp) {
  Value current = argument;
  while (true) {
    // Follow the chain to its end. `tensor()` is null on the operations that
    // consume a tensor without producing one, so the last non-null value is
    // the one the terminating operation takes.
    Value last = current;
    Operation* lastOp = nullptr;
    for (TensorIterator it(cast<TypedValue<RankedTensorType>>(current));
         it != std::default_sentinel; ++it) {
      if (Value currentTensor = it.tensor()) {
        last = currentTensor;
      }
      lastOp = it.operation();
    }

    if (isa_and_nonnull<func::ReturnOp>(lastOp)) {
      for (const auto& [index, operand] :
           llvm::enumerate(returnOp.getOperands())) {
        if (operand == last) {
          return static_cast<int64_t>(index);
        }
      }
      return KEPT;
    }

    // The chain stops at a nested call. Step over it to the result that
    // continues the tensor and keep following from there. Each hop moves
    // forward along the def-use chain, so this terminates.
    auto callOp = dyn_cast_or_null<func::CallOp>(lastOp);
    if (!callOp) {
      return KEPT;
    }
    auto next = getResultForOperand(callOp, last);
    if (failed(next)) {
      return failure();
    }
    if (!*next) {
      return KEPT;
    }
    current = *next;
  }
}

FailureOr<SmallVector<int64_t>>
CallTensorMapping::computeMapping(func::FuncOp callee) {
  if (callee.isExternal()) {
    return failure();
  }

  // Threading a callee already in progress would not terminate.
  if (!inProgress.insert(callee.getOperation()).second) {
    return failure();
  }
  const llvm::scope_exit progressGuard(
      [&] { inProgress.erase(callee.getOperation()); });

  // A body under construction may not have a terminator yet.
  if (!callee.getBody().hasOneBlock() ||
      !callee.getBody().front().mightHaveTerminator()) {
    return failure();
  }
  auto returnOp =
      dyn_cast<func::ReturnOp>(callee.getBody().front().getTerminator());
  if (!returnOp) {
    return failure();
  }

  SmallVector<int64_t> mapping;
  for (BlockArgument arg : callee.getArguments()) {
    if (!isQubitTensor(arg.getType())) {
      continue;
    }
    auto result = threadToResult(arg, returnOp);
    if (failed(result)) {
      return failure();
    }
    mapping.emplace_back(*result);
  }

  return mapping;
}

FailureOr<ArrayRef<int64_t>>
CallTensorMapping::mappingFor(func::CallOp callOp) {
  auto callee = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr()));
  if (!callee) {
    return failure();
  }

  auto* const key = callee.getOperation();
  if (const auto it = cache.find(key); it != cache.end()) {
    return ArrayRef<int64_t>(it->second);
  }
  // Compute before caching so recursion is detected through inProgress.
  auto mapping = computeMapping(callee);
  if (failed(mapping)) {
    return failure();
  }
  return ArrayRef<int64_t>(
      cache.insert_or_assign(key, std::move(*mapping)).first->second);
}

FailureOr<Value> CallTensorMapping::getResultForOperand(func::CallOp callOp,
                                                        Value operand) {
  const auto position = tensorPositionIn(callOp.getOperands(), operand);
  assert(position && "expected a qubit-tensor operand of the call");
  auto mappingOr = mappingFor(callOp);
  if (failed(mappingOr)) {
    return failure();
  }
  ArrayRef<int64_t> mapping = *mappingOr;
  assert(*position < mapping.size() && "expected matching call signature");
  const auto resultIndex = mapping[*position];
  if (resultIndex == KEPT) {
    return Value{};
  }
  return callOp.getResult(static_cast<unsigned>(resultIndex));
}

} // namespace mlir::qtensor
