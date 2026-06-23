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
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <iterator>

namespace mlir::qco {

bool WireIterator::isSinkLikeOperation(Operation* op) {
  return isa<SinkOp, YieldOp, qtensor::InsertOp, scf::YieldOp>(op);
}

bool WireIterator::isSourceLikeOperation(Operation* op) {
  return isa<AllocOp, StaticOp, qtensor::ExtractOp>(op);
}

Value WireIterator::qubit() const {
  if (op_ != nullptr && isSinkLikeOperation(op_)) {
    return nullptr;
  }

  return qubit_;
}

void WireIterator::forward() {
  // If the iterator is a sentinel already, there is nothing to do.
  if (isSentinel_) {
    return;
  }

  // After the final operation comes the sentinel.
  if (isFinal_) {
    isSentinel_ = true;
    return;
  }

  // Find the user-operation of the qubit SSA value.
  assert(qubit_.hasOneUse() && "expected linear typing");
  op_ = *(qubit_.user_begin());

  if (isSinkLikeOperation(op_)) {
    isFinal_ = true;
    return;
  }

  if (!isSourceLikeOperation(op_)) {
    // Find the output from the input qubit SSA value.
    TypeSwitch<Operation*>(op_)
        .Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
          qubit_ = op.getOutputForInput(qubit_);
        })
        .Case<MeasureOp>([&](MeasureOp op) { qubit_ = op.getQubitOut(); })
        .Case<ResetOp>([&](ResetOp op) { qubit_ = op.getQubitOut(); })
        .Case<scf::ForOp, scf::WhileOp>([&](auto op) {
          qubit_ = op.getTiedLoopResult(&*(qubit_.use_begin()));
        })
        .Case<qco::IfOp>([&](qco::IfOp op) {
          auto it = llvm::find(op.getQubits(), qubit_);
          assert(it != op.getQubits().end());
          const auto idx = std::distance(op.getQubits().begin(), it);
          qubit_ = op.getResults()[idx];
        })
        .Default([&](Operation* op) {
          llvm::reportFatalInternalError("unknown op in def-use chain: " +
                                         op->getName().getStringRef());
        });
  }
}

void WireIterator::backward() {
  // If the iterator is a sentinel, reactivate the iterator.
  if (isSentinel_) {
    isSentinel_ = false;
    isFinal_ = true;
    return;
  }

  // If the op is a nullptr, the qubit value is a block argument and thus the
  // beginning of the qubit wire.
  if (op_ == nullptr) {
    return;
  }

  // For these operations, qubit_ is an OpOperand. Hence, only get the def-op.
  if (isSinkLikeOperation(op_)) {
    op_ = qubit_.getDefiningOp();
    isFinal_ = false;
    return;
  }

  // Source-like ops define the start of the qubit wire.
  // Consequently, stop and early exit.
  if (isSourceLikeOperation(op_)) {
    return;
  }

  // Find the input from the output qubit SSA value.
  TypeSwitch<Operation*>(op_)
      .Case<UnitaryOpInterface>(
          [&](UnitaryOpInterface op) { qubit_ = op.getInputForOutput(qubit_); })
      .Case<MeasureOp>([&](MeasureOp op) { qubit_ = op.getQubitIn(); })
      .Case<ResetOp>([&](ResetOp op) { qubit_ = op.getQubitIn(); })
      .Case<scf::ForOp, scf::WhileOp>([&](auto op) {
        if (auto res = dyn_cast<OpResult>(qubit_)) {
          OpOperand* operand = op.getTiedLoopInit(res);
          qubit_ = operand->get();
          return;
        }

        llvm::reportFatalInternalError(
            "expected scf.for result for tied init lookup");
      })
      .Case<qco::IfOp>([&](qco::IfOp op) {
        if (auto res = dyn_cast<OpResult>(qubit_)) {
          auto it = llvm::find(op.getResults(), res);
          assert(it != op->result_end());
          const auto idx = std::distance(op.result_begin(), it);
          qubit_ = op.getQubits()[idx];
          return;
        }

        llvm::reportFatalInternalError(
            "expected scf.for result for tied init lookup");
      })
      .Default([&](Operation* op) {
        llvm::reportFatalInternalError("unknown op in def-use chain: " +
                                       op->getName().getStringRef());
      });

  // Get the operation that produces the qubit value.
  // If the current qubit SSA value is a BlockArgument (no defining op), the
  // operation will be a nullptr.
  op_ = qubit_.getDefiningOp();
  isFinal_ = false;
}

static_assert(std::bidirectional_iterator<WireIterator>);
static_assert(std::sentinel_for<std::default_sentinel_t, WireIterator>,
              "std::default_sentinel_t must be a sentinel for WireIterator.");
} // namespace mlir::qco
