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
#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
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

// Returns the position of a qubit among the qubit-typed values in a range.
template <typename RangeT>
static std::optional<size_t> qubitPositionIn(RangeT range, Value qubit) {
  size_t position = 0;
  for (Value value : range) {
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

// Returns the qubit-typed value at a position, or null if none exists.
template <typename RangeT>
static Value nthQubitOf(RangeT range, size_t position) {
  size_t seen = 0;
  for (Value value : range) {
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

FailureOr<SmallVector<int64_t>>
CallQubitMapping::computeMapping(func::FuncOp callee) {
  if (callee.isExternal()) {
    return failure();
  }

  // Threading a callee already in progress would not terminate.
  if (!inProgress.insert(callee.getOperation()).second) {
    return failure();
  }
  auto progressGuard =
      llvm::make_scope_exit([&] { inProgress.erase(callee.getOperation()); });

  // A body under construction may not have a terminator yet.
  if (!callee.getBody().hasOneBlock() ||
      !callee.getBody().front().mightHaveTerminator()) {
    return failure();
  }
  auto returnOp =
      dyn_cast<func::ReturnOp>(callee.getBody().front().getTerminator());
  if (!returnOp) {
    return failure();
  }

  SmallVector<int64_t> mapping;
  for (BlockArgument arg : callee.getArguments()) {
    if (!isa<QubitType>(arg.getType())) {
      continue;
    }

    int64_t resultIndex = KEPT;
    {
      // Follow the argument to the end of its wire.
      Value last = arg;
      Operation* lastOp = nullptr;
      WireIterator it(arg, this);
      for (; it != std::default_sentinel; ++it) {
        last = it.qubit();
        lastOp = it.operation();
      }
      if (it.mappingFailed_) {
        return failure();
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

  return mapping;
}

void CallQubitMapping::invalidate() { cache.clear(); }

FailureOr<ArrayRef<int64_t>> CallQubitMapping::mappingFor(func::CallOp callOp) {
  auto callee = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr()));
  if (!callee) {
    return failure();
  }

  auto* const key = callee.getOperation();
  if (const auto it = cache.find(key); it != cache.end()) {
    return ArrayRef<int64_t>(it->second);
  }
  // Compute before caching so recursion is detected through inProgress.
  auto mapping = computeMapping(callee);
  if (failed(mapping)) {
    return failure();
  }
  return ArrayRef<int64_t>(
      cache.insert_or_assign(key, std::move(*mapping)).first->second);
}

FailureOr<Value> CallQubitMapping::getResultForOperand(func::CallOp callOp,
                                                       Value operand) {
  const auto position = qubitPositionIn(callOp.getOperands(), operand);
  assert(position && "expected a qubit operand of the call");
  auto mappingOr = mappingFor(callOp);
  if (failed(mappingOr)) {
    return failure();
  }
  ArrayRef<int64_t> mapping = *mappingOr;
  assert(*position < mapping.size() && "expected matching call signature");
  const auto resultIndex = mapping[*position];
  if (resultIndex == KEPT) {
    return Value{};
  }
  return callOp.getResult(static_cast<unsigned>(resultIndex));
}

FailureOr<Value> CallQubitMapping::getOperandForResult(func::CallOp callOp,
                                                       Value result) {
  auto opResult = cast<OpResult>(result);
  assert(opResult.getOwner() == callOp.getOperation() &&
         "expected a result of the call");
  const auto resultIndex = static_cast<int64_t>(opResult.getResultNumber());
  auto mappingOr = mappingFor(callOp);
  if (failed(mappingOr)) {
    return failure();
  }
  ArrayRef<int64_t> mapping = *mappingOr;
  for (const auto& [position, index] : llvm::enumerate(mapping)) {
    if (index == resultIndex) {
      return nthQubitOf(callOp.getOperands(), position);
    }
  }
  return Value{};
}

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

FailureOr<Value> WireIterator::resultForOperand(func::CallOp callOp,
                                                Value operand) const {
  CallQubitMapping local;
  auto& mapping = mapping_ == nullptr ? local : *mapping_;
  return mapping.getResultForOperand(callOp, operand);
}

Value WireIterator::operandForResult(func::CallOp callOp, Value result) const {
  CallQubitMapping local;
  auto& mapping = mapping_ == nullptr ? local : *mapping_;
  auto operand = mapping.getOperandForResult(callOp, result);
  return succeeded(operand) ? *operand : Value{};
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
      .Case<UnitaryOpInterface>(
          [&](UnitaryOpInterface op) { qubit_ = op.getOutputForInput(qubit_); })
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
      .Case<func::CallOp>([&](func::CallOp op) {
        // A call threads the qubit through to the matching result. When the
        // callee keeps it, the wire ends here.

        auto result = resultForOperand(op, qubit_);
        if (failed(result)) {
          mappingFailed_ = true;
          pos_ = Position::Tail;
          return;
        }
        if (!*result) {
          pos_ = Position::Tail;
          return;
        }
        qubit_ = *result;
      })
      .Default([&](Operation*) {
        mappingFailed_ = true;
        pos_ = Position::Tail;
      });
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
        .Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
          qubit_ = op.getInputForOutput(qubit_);
        })
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
        .Case<func::CallOp>([&](func::CallOp callOp) {
          Value operand = operandForResult(callOp, qubit_);
          if (!operand) {
            unknown = true;
            return;
          }
          qubit_ = operand;
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
