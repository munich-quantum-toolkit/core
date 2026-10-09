/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Utils/FunctionUtils.h"

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::qco;

SmallVector<unsigned> mlir::qco::getQuantumArgumentIndices(TypeRange types) {
  SmallVector<unsigned> arguments;
  for (auto [index, type] : llvm::enumerate(types)) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (isa<QubitType>(type) ||
        (tensor && isa<QubitType>(tensor.getElementType()))) {
      arguments.emplace_back(index);
    }
  }
  return arguments;
}

FailureOr<unsigned> mlir::qco::getCallArgumentForResult(func::CallOp call,
                                                        unsigned result) {
  if (!SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr())) {
    return failure();
  }
  auto arguments = getQuantumArgumentIndices(call.getOperandTypes());
  if (call.getNumResults() < arguments.size()) {
    return failure();
  }
  const auto first = call.getNumResults() - arguments.size();
  if (result < first || result >= call.getNumResults()) {
    return failure();
  }
  const auto argument = arguments[result - first];
  if (call.getResult(result).getType() != call.getOperand(argument).getType()) {
    return failure();
  }
  return argument;
}

FailureOr<unsigned> mlir::qco::getCallResultForArgument(func::CallOp call,
                                                        unsigned argument) {
  if (!SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr())) {
    return failure();
  }
  auto arguments = getQuantumArgumentIndices(call.getOperandTypes());
  const auto* position = llvm::find(arguments, argument);
  if (position == arguments.end() || call.getNumResults() < arguments.size()) {
    return failure();
  }
  const auto result = call.getNumResults() - arguments.size() +
                      static_cast<unsigned>(position - arguments.begin());
  if (call.getResult(result).getType() != call.getOperand(argument).getType()) {
    return failure();
  }
  return result;
}

/// Return the terminator of @p block, or null if it has none.
///
/// Verifiers trace values through functions that operation verification may
/// not have reached yet, so a missing terminator must not abort.
[[nodiscard]] static Operation* getTerminatorOrNull(Block& block) {
  return block.mightHaveTerminator() ? block.getTerminator() : nullptr;
}

/// Return whether every block of @p op yields its linear result @p result from
/// the block argument tied to it.
///
/// `qco.if` and `qco.index_switch` tie linear operand `i` to block argument
/// `i` of each region and to linear result `i`; their yields list classical
/// values first, like the results. Tying a result to its operand is only sound
/// when no branch exchanges the values it yields.
[[nodiscard]] static bool yieldsPositionally(Operation* op, OpResult result,
                                             unsigned linearIndex) {
  return llvm::all_of(op->getRegions(), [&](Region& region) {
    if (!region.hasOneBlock()) {
      return false;
    }
    Block& block = region.front();
    auto* terminator = getTerminatorOrNull(block);
    if (terminator == nullptr ||
        result.getResultNumber() >= terminator->getNumOperands()) {
      return false;
    }
    auto argument = traceQubitArgument(
        block, terminator->getOperand(result.getResultNumber()));
    return succeeded(argument) && *argument == linearIndex;
  });
}

FailureOr<Value> mlir::qco::traceQuantumOrigin(Value value) {
  // Unverified IR can contain an SSA cycle, in a graph region or in a block
  // whose dominance has not been checked yet. Every step has a single
  // predecessor, so a repeated value means the trace would never end.
  DenseSet<Value> visited;
  while (true) {
    if (!visited.insert(value).second) {
      return failure();
    }
    auto result = dyn_cast<OpResult>(value);
    if (!result) {
      return value;
    }
    auto* op = result.getOwner();

    if (auto call = dyn_cast<func::CallOp>(op)) {
      auto argument = getCallArgumentForResult(call, result.getResultNumber());
      if (failed(argument)) {
        return failure();
      }
      value = call.getOperand(*argument);
      continue;
    }

    if (auto unitary = dyn_cast<UnitaryOpInterface>(op)) {
      value = unitary.getInputForOutput(value);
      if (!value) {
        return failure();
      }
      continue;
    }

    if (auto measure = dyn_cast<MeasureOp>(op)) {
      if (value != measure.getQubitOut()) {
        return failure();
      }
      value = measure.getQubitIn();
      continue;
    }

    if (auto reset = dyn_cast<ResetOp>(op)) {
      value = reset.getQubitIn();
      continue;
    }

    if (auto extract = dyn_cast<qtensor::ExtractOp>(op)) {
      // An extracted element starts a wire of its own.
      if (value != extract.getOutTensor()) {
        return value;
      }
      value = extract.getTensor();
      continue;
    }

    if (auto insert = dyn_cast<qtensor::InsertOp>(op)) {
      value = insert.getDest();
      continue;
    }

    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      auto* body = loop.getBody();
      auto* yield = getTerminatorOrNull(*body);
      const auto index = result.getResultNumber();
      auto iterArguments = loop.getRegionIterArgs();
      if (yield == nullptr || index >= yield->getNumOperands() ||
          index >= iterArguments.size() || index >= loop.getInitArgs().size()) {
        return failure();
      }
      auto argument = traceQubitArgument(*body, yield->getOperand(index));
      if (failed(argument) ||
          *argument != iterArguments[index].getArgNumber()) {
        return failure();
      }
      value = loop.getInitArgs()[index];
      continue;
    }

    if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      // The result leaves through the condition from some before-argument,
      // and the after-region has to hand it back to that same argument, or
      // the correspondence would change from one iteration to the next.
      const auto index = result.getResultNumber();
      auto condition = dyn_cast_or_null<scf::ConditionOp>(
          getTerminatorOrNull(*loop.getBeforeBody()));
      auto yield = dyn_cast_or_null<scf::YieldOp>(
          getTerminatorOrNull(*loop.getAfterBody()));
      if (!condition || !yield || index >= condition.getArgs().size()) {
        return failure();
      }
      auto before =
          traceQubitArgument(*loop.getBeforeBody(), condition.getArgs()[index]);
      if (failed(before) || *before >= yield.getResults().size() ||
          *before >= loop.getInits().size()) {
        return failure();
      }
      auto after =
          traceQubitArgument(*loop.getAfterBody(), yield.getResults()[*before]);
      if (failed(after) || *after != index) {
        return failure();
      }
      value = loop.getInits()[*before];
      continue;
    }

    if (auto ifOp = dyn_cast<IfOp>(op)) {
      auto* qubit = ifOp.getTiedQubit(result);
      if (qubit == nullptr ||
          !yieldsPositionally(op, result, qubit->getOperandNumber() - 1)) {
        return failure();
      }
      value = qubit->get();
      continue;
    }

    if (auto switchOp = dyn_cast<IndexSwitchOp>(op)) {
      auto* target = switchOp.getTiedTarget(result);
      if (target == nullptr ||
          !yieldsPositionally(op, result, target->getOperandNumber() - 1)) {
        return failure();
      }
      value = target->get();
      continue;
    }

    if (isa<AllocOp, StaticOp, qtensor::AllocOp, qtensor::FromElementsOp>(op)) {
      return value;
    }

    // Any other operation is not known to continue a quantum value.
    return failure();
  }
}

FailureOr<unsigned> mlir::qco::traceQubitArgument(Block& block, Value value) {
  auto origin = traceQuantumOrigin(value);
  if (failed(origin)) {
    return failure();
  }
  auto argument = dyn_cast<BlockArgument>(*origin);
  if (!argument || argument.getOwner() != &block ||
      getQuantumArgumentIndices(TypeRange{argument.getType()}).empty()) {
    return failure();
  }
  return argument.getArgNumber();
}

FailureOr<unsigned> mlir::qco::traceQubitArgument(func::FuncOp function,
                                                  Value value) {
  if (function.isDeclaration()) {
    return failure();
  }
  return traceQubitArgument(function.getBody().front(), value);
}

Operation* mlir::qco::findReleaseInBlock(Value created) {
  Block* block = created.getParentBlock();
  Value value = created;
  // The walk checks linearity itself because the wire and tensor iterators
  // assume it, and verifiers see unverified IR.
  while (value.hasOneUse()) {
    OpOperand& use = *value.use_begin();
    Operation* user = use.getOwner();
    if (user->getBlock() != block) {
      return nullptr;
    }
    if (isa<SinkOp, qtensor::DeallocOp>(user)) {
      // Region operations and calls were crossed by position. Tracing the
      // released value back proves the correspondence that assumed.
      auto origin = traceQuantumOrigin(value);
      return succeeded(origin) && *origin == created ? user : nullptr;
    }
    value = TypeSwitch<Operation*, Value>(user)
                .Case([&](UnitaryOpInterface op) {
                  return op.getOutputForInput(value);
                })
                .Case([](MeasureOp op) { return op.getQubitOut(); })
                .Case([](ResetOp op) { return op.getQubitOut(); })
                .Case([](qtensor::ExtractOp op) { return op.getOutTensor(); })
                .Case([&](qtensor::InsertOp op) {
                  return value == op.getDest() ? op.getResult() : Value{};
                })
                .Case([&](scf::ForOp op) { return op.getTiedLoopResult(&use); })
                .Case([&](scf::WhileOp op) {
                  const auto index = use.getOperandNumber();
                  return index < op->getNumResults() ? op->getResult(index)
                                                     : Value{};
                })
                .Case([&](IfOp op) { return op.getTiedResult(&use); })
                .Case([&](IndexSwitchOp op) { return op.getTiedResult(&use); })
                .Case([&](func::CallOp op) {
                  auto result =
                      getCallResultForArgument(op, use.getOperandNumber());
                  return succeeded(result) ? op.getResult(*result) : Value{};
                })
                .Default([](Operation*) { return Value{}; });
    if (!value) {
      return nullptr;
    }
  }
  return nullptr;
}

/// Require dynamic tensor slots to be restored before leaving each region.
/// Slot identity and disjoint extractions are program preconditions; known
/// violations and positional region correspondence are checked separately.
bool mlir::qco::hasCompleteTensorLifetime(Value tensor, unsigned depth) {
  /// ponytail: reject deeper nesting; use a worklist if proving
  /// completeness beyond 64 nested regions becomes necessary.
  if (depth == 64) {
    return false;
  }
  const auto isTensor = [](Value value) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    return type && isa<qco::QubitType>(type.getElementType());
  };
  unsigned extracted = 0;
  while (tensor.hasOneUse()) {
    auto* user = *tensor.user_begin();
    if (auto extract = dyn_cast<qtensor::ExtractOp>(user)) {
      ++extracted;
      tensor = extract.getOutTensor();
    } else if (auto insert = dyn_cast<qtensor::InsertOp>(user)) {
      if (extracted == 0) {
        return false;
      }
      --extracted;
      tensor = insert.getResult();
    } else if (isa<scf::ForOp, scf::WhileOp, qco::IfOp, qco::IndexSwitchOp>(
                   user)) {
      if (extracted != 0) {
        return false;
      }
      const auto index =
          llvm::count_if(user->getOperands().take_front(
                             tensor.use_begin()->getOperandNumber()),
                         isTensor);
      for (Region& region : user->getRegions()) {
        auto arguments =
            llvm::filter_to_vector(region.getArguments(), isTensor);
        if (index >= arguments.size() ||
            !hasCompleteTensorLifetime(arguments[index], depth + 1)) {
          return false;
        }
      }
      auto results = llvm::filter_to_vector(user->getResults(), isTensor);
      if (index >= results.size()) {
        return false;
      }
      tensor = results[index];
    } else if (auto call = dyn_cast<func::CallOp>(user)) {
      if (extracted != 0) {
        return false;
      }
      auto arguments = qco::getQuantumArgumentIndices(call.getOperandTypes());
      auto* position =
          llvm::find(arguments, tensor.use_begin()->getOperandNumber());
      if (position == arguments.end() ||
          call.getNumResults() < arguments.size()) {
        return false;
      }
      const auto result = call.getNumResults() - arguments.size() +
                          std::distance(arguments.begin(), position);
      auto argument = qco::getCallArgumentForResult(call, result);
      if (failed(argument)) {
        return false;
      }
      tensor = call.getResult(result);
    } else {
      return isa<qtensor::DeallocOp, qco::YieldOp, scf::YieldOp,
                 scf::ConditionOp, func::ReturnOp>(user) &&
             extracted == 0;
    }
  }
  return false;
}
