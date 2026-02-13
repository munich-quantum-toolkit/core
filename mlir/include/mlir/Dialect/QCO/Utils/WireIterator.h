/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <iterator>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

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

  explicit WireIterator() : op_(nullptr), qubit_(nullptr), isSentinel_(false) {}
  explicit WireIterator(mlir::Value qubit)
      : op_(qubit.getDefiningOp()), qubit_(qubit), isSentinel_(false) {}

  [[nodiscard]] mlir::Value qubit() const { return qubit_; }
  [[nodiscard]] mlir::Operation* operation() const { return op_; }
  [[nodiscard]] mlir::Operation* operator*() const { return operation(); }

  WireIterator& operator++() {
    forward();
    return *this;
  }

  WireIterator operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  WireIterator& operator--() {
    backward();
    return *this;
  }

  WireIterator operator--(int) {
    auto tmp = *this;
    --*this;
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
  void forward() {
    // If the iterator is a sentinel already, there is nothing to do.
    if (isSentinel_) {
      return;
    }

    // For dynamic qubits, a deallocation operation defines the end of the qubit
    // wire.
    if (mlir::isa<qco::DeallocOp>(op_)) {
      isSentinel_ = true;
      return;
    }

    if (!(mlir::isa<qco::AllocOp>(op_) || mlir::isa<qco::StaticOp>(op_))) {
      mlir::TypeSwitch<mlir::Operation*>(op_)
          .Case<qco::UnitaryOpInterface>([&](qco::UnitaryOpInterface op) {
            qubit_ = op.getOutputForInput(qubit_);
          })
          .Case<qco::MeasureOp>(
              [&](qco::MeasureOp op) { qubit_ = op.getQubitOut(); })
          .Case<qco::ResetOp>(
              [&](qco::ResetOp op) { qubit_ = op.getQubitOut(); })
          .Default([&](mlir::Operation* op) {
            report_fatal_error("unknown op in def-use chain: " +
                               op->getName().getStringRef());
          });
    }

    // For static qubits, if there are no more uses of the qubit SSA value, the
    // end of the qubit wire is reached.
    if (qubit_.use_empty()) {
      isSentinel_ = true;
      return;
    }

    // Finally, find the user-operation of the qubit SSA value.
    assert(qubit_.getNumUses() == 1);
    op_ = *(qubit_.getUsers().begin());
  }

  /// @brief Move to the previous operation on the qubit wire.
  void backward() {}

  mlir::Operation* op_;
  mlir::Value qubit_;
  bool isSentinel_;
};

static_assert(std::bidirectional_iterator<WireIterator>);
static_assert(std::sentinel_for<std::default_sentinel_t, WireIterator>,
              "std::default_sentinel_t must be a sentinel for WireIterator.");
}; // namespace mlir::qco
