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

#include <iterator>
#include <mlir/IR/Operation.h>

namespace mlir::qco {

/**
 * @brief A bidirectional_iterator traversing the def-use chain of a qubit wire.
 *
 * The iterator follows the flow of a qubit through a sequence of quantum
 * operations while respecting the semantics of the respective operation.
 **/
class [[nodiscard]] WireIterator {
public:
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = mlir::Operation*;

  WireIterator() : op_(nullptr), qubit_(nullptr), isSentinel_(false) {}
  explicit WireIterator(mlir::Value qubit)
      : op_(qubit.getDefiningOp()), qubit_(qubit), isSentinel_(false) {}

  /// @returns the operation the iterator points to.
  [[nodiscard]] mlir::Operation* operation() const { return op_; }

  /// @returns the operation the iterator points to.
  [[nodiscard]] mlir::Operation* operator*() const { return operation(); }

  /// @returns the qubit the iterator points to.
  [[nodiscard]] mlir::Value qubit() const;

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
    return other.qubit_ == qubit_ && other.op_ == op_ &&
           other.isSentinel_ == isSentinel_;
  }

  bool operator==([[maybe_unused]] std::default_sentinel_t s) const {
    return isSentinel_;
  }

private:
  /// @brief Move to the next operation on the qubit wire.
  void forward();

  /// @brief Move to the previous operation on the qubit wire.
  void backward();

  mlir::Operation* op_;
  mlir::Value qubit_;
  bool isSentinel_;
};
} // namespace mlir::qco
