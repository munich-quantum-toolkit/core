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
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

namespace mlir::qco {

/// Return the position of @p qubit among the qubit-typed values of @p range, or
/// `std::nullopt` when it is not one of them.
template <typename RangeT>
static std::optional<size_t> qubitPositionIn(RangeT range, Value qubit) {
  size_t position = 0;
  for (const Value value : range) {
    if (!isa<QubitType>(value.getType())) {
      continue;
    }
    if (value == qubit) {
      return position;
    }
    ++position;
  }
  return std::nullopt;
}

/// Return the qubit-typed value at position @p position of @p range, or a null
/// value when the range holds fewer qubits than that.
template <typename RangeT>
static Value nthQubitOf(RangeT range, size_t position) {
  size_t seen = 0;
  for (const Value value : range) {
    if (!isa<QubitType>(value.getType())) {
      continue;
    }
    if (seen == position) {
      return value;
    }
    ++seen;
  }
  return nullptr;
}

SmallVector<int64_t> CallQubitMapping::positionalMapping(func::FuncOp callee) {
  SmallVector<int64_t> qubitResults;
  for (const auto& [index, type] : llvm::enumerate(callee.getResultTypes())) {
    if (isa<QubitType>(type)) {
      qubitResults.emplace_back(static_cast<int64_t>(index));
    }
  }

  SmallVector<int64_t> mapping;
  size_t seen = 0;
  for (const Type type : callee.getArgumentTypes()) {
    if (!isa<QubitType>(type)) {
      continue;
    }
    mapping.emplace_back(seen < qubitResults.size() ? qubitResults[seen]
                                                    : KEPT);
    ++seen;
  }
  return mapping;
}

SmallVector<int64_t> CallQubitMapping::computeMapping(func::FuncOp callee) {
  // Without a body there is nothing to thread, so the positional pairing is the
  // only contract available.
  if (callee.isExternal()) {
    return positionalMapping(callee);
  }

  // Threading a callee that is already being threaded would not terminate.
  if (!inProgress.insert(callee.getOperation()).second) {
    return positionalMapping(callee);
  }

  // Threading follows a single straight-line body. A callee whose control flow
  // ends somewhere other than one `func.return` cannot be threaded, and
  // claiming that it keeps every qubit would be worse than admitting that only
  // the positional contract is known.
  auto returnOp =
      dyn_cast<func::ReturnOp>(callee.getBody().front().getTerminator());
  if (!callee.getBody().hasOneBlock() || !returnOp) {
    inProgress.erase(callee.getOperation());
    return positionalMapping(callee);
  }

  SmallVector<int64_t> mapping;
  for (const BlockArgument arg : callee.getArguments()) {
    if (!isa<QubitType>(arg.getType())) {
      continue;
    }

    int64_t resultIndex = KEPT;
    {
      // Follow the argument to the end of its wire. `qubit()` is null on the
      // terminating operation, so the last non-null value is the one that
      // operation consumes.
      Value last = arg;
      Operation* lastOp = nullptr;
      for (WireIterator it(arg, this); it != std::default_sentinel; ++it) {
        if (const Value current = it.qubit()) {
          last = current;
        }
        lastOp = it.operation();
      }

      if (isa_and_nonnull<func::ReturnOp>(lastOp)) {
        for (const auto& [index, operand] :
             llvm::enumerate(returnOp.getOperands())) {
          if (operand == last) {
            resultIndex = static_cast<int64_t>(index);
            break;
          }
        }
      }
    }
    mapping.emplace_back(resultIndex);
  }

  inProgress.erase(callee.getOperation());
  return mapping;
}

void CallQubitMapping::invalidate() { cache.clear(); }

ArrayRef<int64_t> CallQubitMapping::mappingFor(func::CallOp callOp) {
  auto callee = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr()));
  if (!callee) {
    return {};
  }

  auto* const key = callee.getOperation();
  if (const auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  // Compute first: the recursion below may insert into the cache itself.
  auto mapping = computeMapping(callee);
  return cache.insert_or_assign(key, std::move(mapping)).first->second;
}

Value CallQubitMapping::getResultForOperand(func::CallOp callOp,
                                            Value operand) {
  const auto position = qubitPositionIn(callOp.getOperands(), operand);
  if (!position) {
    return nullptr;
  }
  const auto mapping = mappingFor(callOp);
  if (*position >= mapping.size()) {
    return nullptr;
  }
  const auto resultIndex = mapping[*position];
  if (resultIndex == KEPT) {
    return nullptr;
  }
  return callOp.getResult(static_cast<unsigned>(resultIndex));
}

Value CallQubitMapping::getOperandForResult(func::CallOp callOp, Value result) {
  const auto opResult = dyn_cast<OpResult>(result);
  if (!opResult || opResult.getOwner() != callOp.getOperation()) {
    return nullptr;
  }
  const auto resultIndex = static_cast<int64_t>(opResult.getResultNumber());
  const auto mapping = mappingFor(callOp);
  for (const auto& [position, index] : llvm::enumerate(mapping)) {
    if (index == resultIndex) {
      return nthQubitOf(callOp.getOperands(), position);
    }
  }
  return nullptr;
}

bool WireIterator::isSinkLikeOperation(Operation* op) {
  return isa<SinkOp, YieldOp, qtensor::InsertOp, scf::ConditionOp, scf::YieldOp,
             func::ReturnOp>(op);
}

bool WireIterator::isSourceLikeOperation(Operation* op) {
  return isa<AllocOp, StaticOp, qtensor::ExtractOp>(op);
}

// Without a shared mapping each query below threads the callee afresh, and the
// nested calls met while doing so are threaded in turn, so the cost grows with
// the depth of the call graph rather than staying constant per step. The
// throw-away mapping is deliberately a local rather than a member, so that
// copying an iterator stays cheap.

Value WireIterator::resultForOperand(func::CallOp callOp, Value operand) const {
  if (mapping_ != nullptr) {
    return mapping_->getResultForOperand(callOp, operand);
  }
  CallQubitMapping local;
  return local.getResultForOperand(callOp, operand);
}

Value WireIterator::operandForResult(func::CallOp callOp, Value result) const {
  if (mapping_ != nullptr) {
    return mapping_->getOperandForResult(callOp, result);
  }
  CallQubitMapping local;
  return local.getOperandForResult(callOp, result);
}

bool WireIterator::atWireStart() const {
  if (op_ == nullptr) {
    return true;
  }
  if (isSourceLikeOperation(op_)) {
    return true;
  }
  // A call is the start of the wire only when it creates the qubit rather than
  // threading one through.
  if (auto callOp = dyn_cast<func::CallOp>(op_)) {
    return qubit_.getDefiningOp() == op_ &&
           operandForResult(callOp, qubit_) == nullptr;
  }
  return false;
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

  // A call threads the qubit through to the matching result. When the callee
  // keeps it, the wire ends here.
  if (auto callOp = dyn_cast<func::CallOp>(op_)) {
    const auto result = resultForOperand(callOp, qubit_);
    if (!result) {
      isFinal_ = true;
      return;
    }
    qubit_ = result;
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
        .Case<scf::ForOp>([&](scf::ForOp op) {
          qubit_ = op.getTiedLoopResult(qubit_.use_begin().getOperand());
        })
        .Case<scf::WhileOp>([&](scf::WhileOp op) {
          // Because the scf::WhileOp doesn't implement "getLoopResults", we
          // have to fallback to the following instead of using
          // "getTiedLoopResult".

          OpOperand* operand = qubit_.use_begin().getOperand();
          qubit_ = op->getResult(operand->getOperandNumber());
        })
        .Case<IfOp>(
            [&](IfOp op) { qubit_ = op.getTiedResult(&(*qubit_.use_begin())); })
        .Case<IndexSwitchOp>([&](IndexSwitchOp op) {
          qubit_ = op.getTiedResult(&(*qubit_.use_begin()));
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

  // A call sits on both sides of a wire: it consumes the caller's qubit and
  // produces a fresh one.
  if (auto callOp = dyn_cast<func::CallOp>(op_)) {
    if (qubit_.getDefiningOp() != op_) {
      // The wire ended at a call that keeps the qubit, so `qubit_` is one of
      // its operands and the previous operation defines it.
      op_ = qubit_.getDefiningOp();
      isFinal_ = false;
      return;
    }
    const auto operand = operandForResult(callOp, qubit_);
    if (!operand) {
      // The callee created the qubit; this is the start of the wire.
      return;
    }
    qubit_ = operand;
    op_ = qubit_.getDefiningOp();
    return;
  }

  // Find the input from the output qubit SSA value.
  TypeSwitch<Operation*>(op_)
      .Case<UnitaryOpInterface>(
          [&](UnitaryOpInterface op) { qubit_ = op.getInputForOutput(qubit_); })
      .Case<MeasureOp>([&](MeasureOp op) { qubit_ = op.getQubitIn(); })
      .Case<ResetOp>([&](ResetOp op) { qubit_ = op.getQubitIn(); })
      .Case<scf::ForOp>([&](scf::ForOp op) {
        if (auto result = dyn_cast<OpResult>(qubit_)) {
          qubit_ = op.getTiedLoopInit(result)->get();
          return;
        }
        llvm::reportFatalInternalError("expected result lookup");
      })
      .Case<scf::WhileOp>([&](scf::WhileOp op) {
        // Because the scf::WhileOp doesn't implement "getLoopResults", we
        // have to fallback to the following instead of using
        // "getTiedLoopInit".

        if (auto result = dyn_cast<OpResult>(qubit_)) {
          qubit_ = op.getInits()[result.getResultNumber()];
          return;
        }

        llvm::reportFatalInternalError("expected result lookup");
      })
      .Case<IfOp>([&](IfOp op) {
        if (auto result = dyn_cast<OpResult>(qubit_)) {
          qubit_ = op.getTiedQubit(result)->get();
          return;
        }
        llvm::reportFatalInternalError("expected result lookup");
      })
      .Case<IndexSwitchOp>([&](IndexSwitchOp op) {
        if (auto result = dyn_cast<OpResult>(qubit_)) {
          qubit_ = op.getTiedTarget(result)->get();
          return;
        }
        llvm::reportFatalInternalError("expected result lookup");
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
