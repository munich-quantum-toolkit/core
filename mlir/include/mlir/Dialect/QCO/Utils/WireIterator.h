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

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <cstdint>
#include <iterator>

namespace mlir::qco {

/// Resolves how qubits flow across call boundaries.
///
/// The mapping follows each qubit argument through the callee instead of
/// assuming positional correspondence. Results are cached per callee. Mapping
/// fails for declarations, recursion, and non-straight-line bodies.
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
  friend class WireIterator;

  // Marks a qubit argument that never reaches a result.
  static constexpr int64_t KEPT = -1;

  // Returns each qubit argument's call-result index, or KEPT.
  FailureOr<ArrayRef<int64_t>> mappingFor(func::CallOp callOp);

  // Derives a mapping by threading every qubit argument through the callee.
  FailureOr<SmallVector<int64_t>> computeMapping(func::FuncOp callee);

  // Gets the call operand feeding a result's wire.
  FailureOr<Value> getOperandForResult(func::CallOp callOp, Value result);

  DenseMap<Operation*, SmallVector<int64_t>> cache;
  DenseSet<Operation*> inProgress;
};

/// A bidirectional iterator over the def-use chain of a qubit wire.
///
/// The iterator follows the flow of a qubit through a sequence of quantum
/// operations while respecting the semantics of each operation.
class [[nodiscard]] WireIterator {
public:
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = Operation*;

  /// Construct a dead-end sentinel wire-iterator.
  WireIterator() : op_(nullptr), qubit_(nullptr), pos_(Position::PastTail) {}

  /// Construct a wire iterator pointing at the defining op of a qubit value.
  explicit WireIterator(Value qubit)
      : op_(qubit.getDefiningOp()), qubit_(qubit) {
    if (op_ == nullptr || isHead(op_)) {
      pos_ = Position::Head;
    } else if (isTail(op_)) {
      pos_ = Position::Tail;
    } else {
      pos_ = Position::Between;
    }
  }

  /// Return the operation the iterator points to.
  [[nodiscard]] Operation* operation() const;

  /// Return the qubit the iterator points to. Terminal operations retain their
  /// input qubit. Sentinel access reports a fatal internal error.
  [[nodiscard]] Value qubit() const;

  /// Return the operation the iterator points to.
  [[nodiscard]] Operation* operator*() const { return operation(); }

  WireIterator& operator++() {
    forward();
    return *this;
  }

  WireIterator operator++(int) {
    auto tmp = *this;
    operator++();
    return tmp;
  }

  WireIterator& operator--() {
    backward();
    return *this;
  }

  WireIterator operator--(int) {
    auto tmp = *this;
    operator--();
    return tmp;
  }

  bool operator==(const WireIterator& other) const {
    return other.qubit_ == qubit_ && other.op_ == op_ && pos_ == other.pos_;
  }

  bool operator==([[maybe_unused]] std::default_sentinel_t s) const {
    return pos_ == Position::PastTail || pos_ == Position::BeforeHead;
  }

private:
  friend class CallQubitMapping;

  /// Labels the position on the wire.
  enum class Position : uint8_t { BeforeHead, Head, Between, Tail, PastTail };

  WireIterator(Value qubit, CallQubitMapping* mapping) : WireIterator(qubit) {
    mapping_ = mapping;
  }

  /// Return true, if an op doesn't return, but only consumes, a qubit value.
  static bool isTail(Operation*);

  /// Return true, if an op doesn't consume, but only returns, a qubit value.
  static bool isHead(Operation*);

  // Moves to the next operation on the qubit wire.
  void forward();

  // Moves to the previous operation on the qubit wire.
  void backward();

  Operation* op_;
  Value qubit_;
  Position pos_;
  bool mappingFailed_ = false;

  // Resolves the call result continuing an operand's wire.
  FailureOr<Value> resultForOperand(func::CallOp callOp, Value operand) const;

  // Resolves the call operand feeding a result's wire.
  [[nodiscard]] Value operandForResult(func::CallOp callOp, Value result) const;

  // Null means that each call query uses a fresh mapping.
  CallQubitMapping* mapping_ = nullptr;
};

/// Categorizes the current traversal direction.
enum class WireDirection : bool { Forward, Backward };

template <WireDirection Direction> struct WireTraversalTraits {
  /// Return the increment stride size.
  static constexpr std::ptrdiff_t stride() {
    if constexpr (Direction == WireDirection::Forward) {
      return 1;
    }
    return -1;
  }
};

/// A range over the def-use chain of a qubit wire.
struct WireRange {
  explicit WireRange(Value qubit) : begin_(qubit) {}

  [[nodiscard]] WireIterator begin() const { return begin_; }
  [[nodiscard]] static std::default_sentinel_t end() {
    return std::default_sentinel;
  }

private:
  WireIterator begin_;
};
} // namespace mlir::qco
