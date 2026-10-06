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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <cstdint>

namespace mlir::qco {

/// Resolves how qubits flow across generic `func.call` boundaries.
///
/// A generic callee may hand its qubits back in any order, or keep them
/// altogether, so the mapping follows each qubit argument through the callee
/// body instead of assuming the positional ABI that `mqt.unitary` functions
/// guarantee. Results are cached per callee. Mapping fails for declarations,
/// recursion, and bodies the wire iterator cannot interpret.
class CallQubitMapping {
public:
  /// Gets the result continuing @p operand's wire.
  ///
  /// Returns a null value when the callee keeps the qubit and failure when the
  /// correspondence cannot be derived.
  [[nodiscard]] FailureOr<Value> getResultForOperand(func::CallOp callOp,
                                                     Value operand);

  /// Clears all cached correspondence after a callee is changed or erased.
  void invalidate();

private:
  // Marks a qubit argument that never reaches a result.
  static constexpr int64_t KEPT = -1;

  // Returns each qubit argument's call-result index, or KEPT.
  FailureOr<ArrayRef<int64_t>> mappingFor(func::CallOp callOp);

  // Derives a mapping by threading every qubit argument through the callee.
  FailureOr<SmallVector<int64_t>> computeMapping(func::FuncOp callee);

  // Follows an argument to a return operand, hopping over nested calls.
  FailureOr<int64_t> threadToResult(Value argument, func::ReturnOp returnOp);

  DenseMap<Operation*, SmallVector<int64_t>> cache;
  DenseSet<Operation*> inProgress;
};

} // namespace mlir::qco
