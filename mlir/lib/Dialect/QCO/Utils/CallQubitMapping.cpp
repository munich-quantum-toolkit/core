/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Utils/CallQubitMapping.h"

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/Utils/WireIterator.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

namespace mlir::qco {

// Returns the position of a qubit among the qubit-typed values in a range.
template <typename RangeT>
static std::optional<size_t> qubitPositionIn(RangeT range, Value qubit) {
  size_t position = 0;
  for (Value value : range) {
    if (!isa<QubitType>(value.getType())) {
      continue;
    }
    if (value == qubit) {
      return position;
    }
    ++position;
  }
  return std::nullopt;
}

FailureOr<int64_t> CallQubitMapping::threadToResult(Value argument,
                                                    func::ReturnOp returnOp) {
  Value current = argument;
  while (true) {
    // Follow the wire to its end. Terminal operations retain their input
    // qubit, so the last value seen is the one the final operation takes.
    Value last = current;
    Operation* lastOp = nullptr;
    for (WireIterator it(current); it != std::default_sentinel; ++it) {
      last = it.qubit();
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

    // The wire stops at a nested call. Step over it to the result that
    // continues the qubit and keep following from there. Each hop moves
    // forward along the def-use chain, so this terminates.
    if (auto callOp = dyn_cast_or_null<func::CallOp>(lastOp)) {
      auto next = getResultForOperand(callOp, last);
      if (failed(next)) {
        return failure();
      }
      if (!*next) {
        return KEPT;
      }
      current = *next;
      continue;
    }

    // An operation that only consumes the qubit keeps it. Anything else is not
    // known to thread the qubit, and guessing would map the argument to a
    // result that does not carry it.
    if (lastOp == nullptr || !WireIterator::isTail(lastOp)) {
      return failure();
    }
    return KEPT;
  }
}

FailureOr<SmallVector<int64_t>>
CallQubitMapping::computeMapping(func::FuncOp callee) {
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
    if (!isa<QubitType>(arg.getType())) {
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

void CallQubitMapping::invalidate() { cache.clear(); }

FailureOr<ArrayRef<int64_t>> CallQubitMapping::mappingFor(func::CallOp callOp) {
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

FailureOr<Value> CallQubitMapping::getResultForOperand(func::CallOp callOp,
                                                       Value operand) {
  const auto position = qubitPositionIn(callOp.getOperands(), operand);
  assert(position && "expected a qubit operand of the call");
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

} // namespace mlir::qco
