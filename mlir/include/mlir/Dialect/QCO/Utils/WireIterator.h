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

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <cstdint>
#include <iterator>

namespace mlir::qco {

/// A bidirectional_iterator traversing the def-use chain of a qubit wire.
///
/// The iterator follows the flow of a qubit through a sequence of quantum
/// operations while respecting the semantics of the respective operation.
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

  /// Return the qubit the iterator points to.
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
  /// Labels the position on the wire.
  enum class Position : uint8_t { BeforeHead, Head, Between, Tail, PastTail };

  /// Return true, if an op doesn't return, but only consumes, a qubit value.
  static bool isTail(Operation*);

  /// Return true, if an op doesn't consume, but only returns, a qubit value.
  static bool isHead(Operation*);

  /// Move to the next operation on the qubit wire.
  void forward();

  /// Move to the previous operation on the qubit wire.
  void backward();

  Operation* op_;
  Value qubit_;
  Position pos_;
};

/// Categorizes the current traversal direction.
enum class WireDirection : bool { Forward, Backward };

template <WireDirection Direction> struct WireTraversalTraits {};

template <> struct WireTraversalTraits<WireDirection::Forward> {
  /// Return the forward increment stride size.
  static constexpr std::ptrdiff_t stride() { return 1; }
};

template <> struct WireTraversalTraits<WireDirection::Backward> {
  /// Return the backward increment stride size.
  static constexpr std::ptrdiff_t stride() { return -1; }
};

/**
 * @brief A range over the def-use chain of a qubit wire, usable in range-based
 * for-loops.
 *
 * Example:
 * @code
 * for (auto* op : WireRange(qubit)) { ... }
 * @endcode
 */
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
