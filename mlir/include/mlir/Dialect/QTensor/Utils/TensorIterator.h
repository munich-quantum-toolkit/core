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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <iterator>

namespace mlir::qtensor {

/// A bidirectional iterator traversing the tensor chain.
class [[nodiscard]] TensorIterator {
public:
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = Operation*;

  TensorIterator()
      : op_(nullptr), tensor_(nullptr), isFinal_(false), isSentinel_(false) {}
  explicit TensorIterator(TypedValue<RankedTensorType> tensor)
      : op_(tensor.getDefiningOp()), tensor_(tensor), isFinal_(false),
        isSentinel_(false) {}

  /// @returns the operation the iterator points to.
  [[nodiscard]] Operation* operation() const { return op_; }

  /// @returns the operation the iterator points to.
  [[nodiscard]] Operation* operator*() const { return operation(); }

  /// @returns the tensor the iterator points to.
  [[nodiscard]] TypedValue<RankedTensorType> tensor() const;

  TensorIterator& operator++() {
    forward();
    return *this;
  }

  TensorIterator operator++(int) {
    auto tmp = *this;
    operator++();
    return tmp;
  }

  TensorIterator& operator--() {
    backward();
    return *this;
  }

  TensorIterator operator--(int) {
    auto tmp = *this;
    operator--();
    return tmp;
  }

  bool operator==(const TensorIterator& other) const {
    return other.tensor_ == tensor_ && other.op_ == op_ &&
           other.isSentinel_ == isSentinel_;
  }

  bool operator==([[maybe_unused]] std::default_sentinel_t s) const {
    return isSentinel_;
  }

private:
  // Moves to the next operation on the tensor def-use chain.
  void forward();

  // Moves to the previous operation on the tensor def-use chain.
  void backward();

  Operation* op_;
  TypedValue<RankedTensorType> tensor_;
  bool isFinal_;
  bool isSentinel_;
};

/// Resolves how qubit tensors flow across call boundaries.
///
/// The mapping follows each tensor argument through the callee instead of
/// assuming positional correspondence. Results are cached per callee. Mapping
/// fails for declarations, recursion, and non-straight-line bodies.
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

  // Follows an argument to a return operand, hopping over calls.
  FailureOr<int64_t> threadToResult(Value arg, func::ReturnOp returnOp);

  DenseMap<Operation*, SmallVector<int64_t>> cache;
  DenseSet<Operation*> inProgress;
};
} // namespace mlir::qtensor
