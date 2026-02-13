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

class [[nodiscard]] WireIterator {
public:
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = mlir::Operation*;

  explicit WireIterator() : op(nullptr), qubit(nullptr), isSentinel(false) {}
  explicit WireIterator(mlir::Value q)
      : op(q.getDefiningOp()), qubit(q), isSentinel(false) {}

  [[nodiscard]] mlir::Operation* operator*() const {
    assert(!sentinel && "Dereferencing sentinel iterator");
    assert(currOp && "Dereferencing null operation");
    return op;
  }

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
    return other.qubit == qubit && other.op == op &&
           other.isSentinel == isSentinel;
  }

  bool operator==([[maybe_unused]] std::default_sentinel_t s) const {
    return isSentinel;
  }

private:
  void forward() {
    // If the iterator is a sentinel already, there is nothing to do.
    if (isSentinel) {
      return;
    }

    // For dynamic qubits, a deallocation operation defines the end of the qubit
    // wire.
    if (mlir::isa<DeallocOp>(op)) {
      isSentinel = true;
    }

    if (!(mlir::isa<qco::AllocOp>(op) || mlir::isa<qco::StaticOp>(op))) {
      mlir::TypeSwitch<mlir::Operation*>(op)
          .Case<qco::UnitaryOpInterface>([&](qco::UnitaryOpInterface op) {
            qubit = op.getOutputForInput(qubit);
          })
          .Case<qco::MeasureOp>(
              [&](qco::MeasureOp op) { qubit = op.getQubitOut(); })
          .Case<qco::ResetOp>(
              [&](qco::ResetOp op) { qubit = op.getQubitOut(); })
          .Default([&](mlir::Operation* op) {
            report_fatal_error("unknown op in def-use chain: " +
                               op->getName().getStringRef());
          });
    }

    // For static qubits, if there are no more uses of the qubit SSA value, the
    // end of the qubit wire is reached.
    if (qubit.use_empty()) {
      isSentinel = true;
      return;
    }

    // Find the user-operation of the qubit SSA value.
    assert(qubit.getNumUses() == 1);
    op = *(qubit.getUsers().begin());
  }

  void backward() {}

  mlir::Operation* op;
  mlir::Value qubit;

  bool isSentinel;
};

static_assert(std::bidirectional_iterator<WireIterator>);
static_assert(std::sentinel_for<std::default_sentinel_t, WireIterator>,
              "std::default_sentinel_t must be a sentinel for WireIterator.");
}; // namespace mlir::qco
