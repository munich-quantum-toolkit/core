/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QTensor/Utils/TensorIterator.h"

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <iterator>

namespace mlir::qtensor {
TypedValue<RankedTensorType> TensorIterator::tensor() const {
  // The following operations don't have an OpResult.
  if (op_ != nullptr && isa<DeallocOp, scf::YieldOp, qco::YieldOp>(op_)) {
    return nullptr;
  }

  return tensor_;
}

void TensorIterator::forward() {
  // If the iterator is a sentinel already, there is nothing to do.
  if (isSentinel_) {
    return;
  }

  // After the final operation comes the sentinel.
  if (isFinal_) {
    isSentinel_ = true;
    return;
  }

  // Find the user-operation of the tensor SSA value.
  assert(tensor_.hasOneUse() && "expected linear typing");
  op_ = *(tensor_.user_begin());

  // The following operations define the end of the tensor's life-chain.
  if (isa<DeallocOp, scf::YieldOp, qco::YieldOp>(op_)) {
    isFinal_ = true;
    return;
  }

  // Find the output from the input tensor SSA value.
  if (!(isa<AllocOp, FromElementsOp>(op_))) {
    TypeSwitch<Operation*>(op_)
        .Case<ExtractOp>([&](ExtractOp op) { tensor_ = op.getOutTensor(); })
        .Case<InsertOp>([&](InsertOp op) { tensor_ = op.getResult(); })
        .Case<scf::ForOp>([&](scf::ForOp op) {
          tensor_ = cast<TypedValue<RankedTensorType>>(
              op.getTiedLoopResult(&*(tensor_.use_begin())));
        })
        .Case<qco::IfOp>([&](qco::IfOp op) {
          auto it = llvm::find(op.getQubits(), tensor_);
          assert(it != op.getQubits().end());
          const auto idx = std::distance(op.getQubits().begin(), it);
          tensor_ = cast<TypedValue<RankedTensorType>>(op.getResults()[idx]);
        })
        .Default([&](Operation* op) {
          report_fatal_error("unknown op in def-use chain: " +
                             op->getName().getStringRef());
        });
  }
}

void TensorIterator::backward() {
  // If the iterator is a sentinel, reactivate the iterator.
  if (isSentinel_) {
    isSentinel_ = false;
    isFinal_ = true;
    return;
  }

  // If the op is a nullptr, the tensor value is a block argument and thus the
  // beginning of the tensor's life-chain.
  if (op_ == nullptr) {
    return;
  }

  // For these operations, tensor_ is an OpOperand. Hence, only get the def-op.
  if (isa<DeallocOp, scf::YieldOp, qco::YieldOp>(op_)) {
    op_ = tensor_.getDefiningOp();
    isFinal_ = false;
    return;
  }

  // Allocations and FromElements define the start of the tensor's life-chain.
  // Consequently, stop and early exit.
  if (isa<AllocOp, FromElementsOp>(op_)) {
    return;
  }

  // Find the input from the output tensor SSA value.
  TypeSwitch<Operation*>(op_)
      .Case<ExtractOp>([&](ExtractOp op) { tensor_ = op.getTensor(); })
      .Case<InsertOp>([&](InsertOp op) { tensor_ = op.getDest(); })
      .Case<scf::ForOp>([&](scf::ForOp op) {
        if (auto res = dyn_cast<OpResult>(tensor_)) {
          OpOperand* operand = op.getTiedLoopInit(res);
          tensor_ = cast<TypedValue<RankedTensorType>>(operand->get());
          return;
        }

        llvm::reportFatalInternalError(
            "expected scf.for result for tied init lookup");
      })
      .Case<qco::IfOp>([&](qco::IfOp op) {
        if (auto res = dyn_cast<OpResult>(tensor_)) {
          auto it = llvm::find(op.getResults(), res);
          assert(it != op->result_end());
          const auto idx = std::distance(op.result_begin(), it);
          tensor_ = cast<TypedValue<RankedTensorType>>(op.getQubits()[idx]);
          return;
        }

        llvm::reportFatalInternalError(
            "expected scf.for result for tied init lookup");
      })
      .Default([&](Operation* op) {
        llvm::reportFatalInternalError("unknown op in def-use chain: " +
                                       op->getName().getStringRef());
      });

  // Get the operation that produces the tensor value.
  // If the current tensor SSA value is a BlockArgument (no defining op), the
  // operation will be a nullptr.
  op_ = tensor_.getDefiningOp();
  isFinal_ = false;
}

static_assert(std::bidirectional_iterator<TensorIterator>);
static_assert(std::sentinel_for<std::default_sentinel_t, TensorIterator>,
              "std::default_sentinel_t must be a sentinel for TensorIterator.");
} // namespace mlir::qtensor
