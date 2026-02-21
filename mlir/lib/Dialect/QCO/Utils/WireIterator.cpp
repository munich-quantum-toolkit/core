/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Utils/WireIterator.h"

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <cassert>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace mlir::qco {
mlir::Value WireIterator::qubit() const {
  // A deallocation doesn't have an OpResult.
  if (op_ != nullptr && mlir::isa<DeallocOp>(op_)) {
    return nullptr;
  }
  return qubit_;
}

void WireIterator::forward() {
  // If the iterator is a sentinel already, there is nothing to do.
  if (isSentinel_) {
    return;
  }

  // Find the user-operation of the qubit SSA value.
  assert(qubit_.getNumUses() == 1 && "expected linear typing");
  op_ = *(qubit_.getUsers().begin());

  // A deallocation op defines the end of the qubit wire (dynamic and static).
  if (mlir::isa<DeallocOp>(op_)) {
    isSentinel_ = true;
    return;
  }

  if (!(mlir::isa<AllocOp, StaticOp>(op_))) {
    // Find the output from the input qubit SSA value.
    mlir::TypeSwitch<mlir::Operation*>(op_)
        .Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
          qubit_ = op.getOutputForInput(qubit_);
        })
        .Case<MeasureOp>([&](MeasureOp op) { qubit_ = op.getQubitOut(); })
        .Case<ResetOp>([&](ResetOp op) { qubit_ = op.getQubitOut(); })
        .Default([&](mlir::Operation* op) {
          report_fatal_error("unknown op in def-use chain: " +
                             op->getName().getStringRef());
        });
  }
}

void WireIterator::backward() {
  // If the iterator is a sentinel, reactivate the iterator.
  if (isSentinel_) {
    isSentinel_ = false;
    return;
  }

  // For deallocations, qubit_ is an OpOperand. Hence, only get the def-op.
  if (mlir::isa<DeallocOp>(op_)) {
    op_ = qubit_.getDefiningOp();
    return;
  }

  // Allocations or static definitions define the start of the qubit wire.
  // Consequently, stop and early exit.
  if (mlir::isa<AllocOp, StaticOp>(op_)) {
    return;
  }

  // Find the input from the output qubit SSA value.
  mlir::TypeSwitch<mlir::Operation*>(op_)
      .Case<UnitaryOpInterface>(
          [&](UnitaryOpInterface op) { qubit_ = op.getInputForOutput(qubit_); })
      .Case<MeasureOp>([&](MeasureOp op) { qubit_ = op.getQubitIn(); })
      .Case<ResetOp>([&](ResetOp op) { qubit_ = op.getQubitIn(); })
      .Default([&](mlir::Operation* op) {
        report_fatal_error("unknown op in def-use chain: " +
                           op->getName().getStringRef());
      });

  // Get the operation that produces the qubit value.
  // If the current qubit SSA value is a BlockArgument (no defining op), stop.
  op_ = qubit_.getDefiningOp();
  if (op_ == nullptr) {
    return;
  }
}
} // namespace mlir::qco
