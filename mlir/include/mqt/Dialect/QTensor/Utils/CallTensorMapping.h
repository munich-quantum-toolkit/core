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
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::qtensor {

/// Resolves how qubit tensors flow across generic `func.call` boundaries.
///
/// Qubit tensors never cross a `qco.call`, so every call carrying one is
/// generic and may hand its tensors back in any order, or keep them. The
/// mapping therefore follows each tensor argument through the callee body.
/// Results are cached per callee. Mapping fails for declarations, recursion,
/// and non-straight-line bodies.
class CallTensorMapping {
public:
  /// Gets the result continuing @p operand's tensor chain.
  ///
  /// Returns a null value when the callee keeps the tensor and failure when the
  /// correspondence cannot be derived.
  [[nodiscard]] FailureOr<Value> getResultForOperand(func::CallOp callOp,
                                                     Value operand);

private:
  // Marks a tensor argument that never reaches a result.
  static constexpr int64_t KEPT = -1;

  // Returns each tensor argument's call-result index, or KEPT.
  FailureOr<ArrayRef<int64_t>> mappingFor(func::CallOp callOp);

  // Derives a mapping by threading every tensor argument through the callee.
  FailureOr<SmallVector<int64_t>> computeMapping(func::FuncOp callee);

  // Follows an argument to a return operand, hopping over nested calls.
  FailureOr<int64_t> threadToResult(Value argument, func::ReturnOp returnOp);

  DenseMap<Operation*, SmallVector<int64_t>> cache;
  DenseSet<Operation*> inProgress;
};

} // namespace mlir::qtensor
