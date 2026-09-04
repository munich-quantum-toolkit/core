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

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstddef>
#include <iterator>

namespace mlir::qco {

bool WireIterator::isTail(Operation* op) {
  // `qtensor.from_elements` takes qubits into a tensor just like
  // `qtensor.insert` does, so a wire reaching either of them ends there.
  return isa<SinkOp, YieldOp, qtensor::InsertOp, qtensor::FromElementsOp,
             scf::ConditionOp, scf::YieldOp, func::ReturnOp>(op);
}

bool WireIterator::isHead(Operation* op) {
  return isa<AllocOp, StaticOp, qtensor::ExtractOp>(op);
}

Operation* WireIterator::operation() const {
  if (*this == std::default_sentinel) {
    llvm::reportFatalInternalError("Trying to access operation of sentinel!");
  }

  return op_;
}

Value WireIterator::qubit() const {
  if (*this == std::default_sentinel) {
    llvm::reportFatalInternalError("Trying to access qubit of sentinel!");
  }

  return qubit_;
}

void WireIterator::forward() {
  // If the iterator is a tail-sentinel already, there is nothing to do.
  if (pos_ == Position::PastTail) {
    return;
  }

  // After the final operation comes the sentinel.
  if (pos_ == Position::Tail) {
    pos_ = Position::PastTail;
    return;
  }

  // If the iterator is a head-sentinel, reactivate the iterator.
  if (pos_ == Position::BeforeHead) {
    pos_ = Position::Head;
    return;
  }

  // Find the user-operation of the qubit SSA value.
  assert(qubit_.hasOneUse() && "expected linear typing");
  op_ = *(qubit_.user_begin());

  if (isTail(op_)) {
    pos_ = Position::Tail;
    return;
  }

  // Find the output from the input qubit SSA value.
  pos_ = Position::Between;

  TypeSwitch<Operation*>(op_)
      .Case(
          [&](UnitaryOpInterface op) { qubit_ = op.getOutputForInput(qubit_); })
      .Case([&](MeasureOp op) { qubit_ = op.getQubitOut(); })
      .Case([&](ResetOp op) { qubit_ = op.getQubitOut(); })
      .Case([&](scf::ForOp op) {
        qubit_ = op.getTiedLoopResult(qubit_.use_begin().getOperand());
      })
      .Case([&](scf::WhileOp op) {
        // Because the scf::WhileOp doesn't implement "getLoopResults", we
        // have to fallback to the following instead of using
        // "getTiedLoopResult".

        OpOperand* operand = qubit_.use_begin().getOperand();
        qubit_ = op->getResult(operand->getOperandNumber());
      })
      .Case([&](IfOp op) { qubit_ = op.getTiedResult(&(*qubit_.use_begin())); })
      .Case([&](IndexSwitchOp op) {
        qubit_ = op.getTiedResult(&(*qubit_.use_begin()));
      })
      .Default([&](Operation*) { pos_ = Position::Tail; });
}

void WireIterator::backward() {
  // If the iterator is a head-sentinel already, there is nothing to do.
  if (pos_ == Position::BeforeHead) {
    return;
  }

  // If the iterator is a tail-sentinel, reactivate the iterator.
  if (pos_ == Position::PastTail) {
    pos_ = Position::Tail;
    return;
  }

  // Before the head operation comes the sentinel.
  if (pos_ == Position::Head) {
    pos_ = Position::BeforeHead;
    return;
  }

  // Head operations are always labeled as Position::Head,
  // so they never reach this point.
  assert((pos_ == Position::Tail || !isHead(op_)) &&
         "expected head ops to carry Position::Head");

  // Tails only consume, not produce values.
  if (pos_ != Position::Tail) {
    bool unknown = false;
    // Find the input from the output qubit SSA value.
    TypeSwitch<Operation*>(op_)
        .Case([&](UnitaryOpInterface op) {
          qubit_ = op.getInputForOutput(qubit_);
        })
        .Case([&](MeasureOp op) { qubit_ = op.getQubitIn(); })
        .Case([&](ResetOp op) { qubit_ = op.getQubitIn(); })
        .Case([&](scf::ForOp op) {
          if (auto result = dyn_cast<OpResult>(qubit_)) {
            qubit_ = op.getTiedLoopInit(result)->get();
            return;
          }
          llvm::reportFatalInternalError("expected result lookup");
        })
        .Case([&](scf::WhileOp op) {
          // Because the scf::WhileOp doesn't implement "getLoopResults", we
          // have to fallback to the following instead of using
          // "getTiedLoopInit".

          if (auto result = dyn_cast<OpResult>(qubit_)) {
            qubit_ = op.getInits()[result.getResultNumber()];
            return;
          }

          llvm::reportFatalInternalError("expected result lookup");
        })
        .Case([&](IfOp op) {
          if (auto result = dyn_cast<OpResult>(qubit_)) {
            qubit_ = op.getTiedQubit(result)->get();
            return;
          }
          llvm::reportFatalInternalError("expected result lookup");
        })
        .Case([&](IndexSwitchOp op) {
          if (auto result = dyn_cast<OpResult>(qubit_)) {
            qubit_ = op.getTiedTarget(result)->get();
            return;
          }
          llvm::reportFatalInternalError("expected result lookup");
        })
        .Default([&](Operation*) { unknown = true; });

    if (unknown) {
      pos_ = Position::BeforeHead;
      return;
    }
  }

  // Get the operation that produces the qubit value.
  // If the current qubit SSA value is a BlockArgument (no defining op), the
  // operation will be a nullptr.
  op_ = qubit_.getDefiningOp();

  if (op_ == nullptr || isHead(op_)) {
    pos_ = Position::Head;
  } else {
    pos_ = Position::Between;
  }
}

static_assert(std::bidirectional_iterator<WireIterator>);
static_assert(std::sentinel_for<std::default_sentinel_t, WireIterator>,
              "std::default_sentinel_t must be a sentinel for WireIterator.");
} // namespace mlir::qco
