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

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <iterator>

namespace mlir::qtensor {

/**
 * @brief A bidirectional_iterator traversing the def-use chain of a tensor.
 **/
class [[nodiscard]] TensorIterator {
public:
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = Operation*;

  TensorIterator() : op_(nullptr), tensor_(nullptr), isSentinel_(false) {}
  explicit TensorIterator(TypedValue<RankedTensorType> tensor)
      : op_(tensor.getDefiningOp()), tensor_(tensor), isSentinel_(false) {}

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
  /// @brief Move to the next operation on the tensor def-use chain.
  void forward();

  /// @brief Move to the previous operation on the tensor def-use chain.
  void backward();

  Operation* op_;
  TypedValue<RankedTensorType> tensor_;
  bool isSentinel_;
};
} // namespace mlir::qco
