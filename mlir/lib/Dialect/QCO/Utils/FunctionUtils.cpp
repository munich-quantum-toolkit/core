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

#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Utils/WireIterator.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"
#include "mqt/Dialect/QTensor/Utils/TensorIterator.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/WalkResult.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <iterator>

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

FailureOr<unsigned> mlir::qco::traceQubitArgument(Block& block, Value value) {
  while (true) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (argument.getOwner() == &block &&
          !getQuantumArgumentIndices(TypeRange{argument.getType()}).empty()) {
        return argument.getArgNumber();
      }
      return failure();
    }

    if (auto call = value.getDefiningOp<func::CallOp>()) {
      auto argument = getCallArgumentForResult(
          call, cast<OpResult>(value).getResultNumber());
      if (failed(argument)) {
        return failure();
      }
      value = call.getOperand(*argument);
      continue;
    }

    if (auto loop = value.getDefiningOp<scf::WhileOp>()) {
      auto result = cast<OpResult>(value).getResultNumber();
      auto argument = traceQubitArgument(
          *loop.getBeforeBody(), loop.getConditionOp().getArgs()[result]);
      if (failed(argument)) {
        return failure();
      }
      value = loop.getInits()[*argument];
      continue;
    }

    if (auto tensor = dyn_cast<TypedValue<RankedTensorType>>(value)) {
      qtensor::TensorIterator iterator(tensor);
      --iterator;
      if (iterator == std::default_sentinel || iterator.tensor() == value) {
        return failure();
      }
      value = iterator.tensor();
    } else {
      WireIterator iterator(value);
      --iterator;
      if (iterator == std::default_sentinel) {
        return failure();
      }
      value = iterator.qubit();
    }
  }
}

FailureOr<unsigned> mlir::qco::traceQubitArgument(func::FuncOp function,
                                                  Value value) {
  if (function.isDeclaration()) {
    return failure();
  }
  return traceQubitArgument(function.getBody().front(), value);
}

bool mlir::qco::hasCompleteTensorLifetime(Value tensor, unsigned depth) {
  // Reject nesting at depth 64 to bound recursive stack use.
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

LogicalResult mlir::qco::collectWireOrigins(Operation* root,
                                            DenseMap<Value, Value>& origins) {
  const auto origin = [&](Value value) {
    auto known = origins.lookup(value);
    return known ? known : value;
  };
  const auto corresponds = [&](ValueRange inputs, ValueRange outputs) {
    if (countIfQubit(inputs) != countIfQubit(outputs)) {
      return false;
    }
    return llvm::all_of(
        llvm::zip_equal(quantumValues(inputs), quantumValues(outputs)),
        [&](auto pair) {
          auto [input, output] = pair;
          return input.getType() == output.getType() &&
                 origin(input) == origin(output);
        });
  };
  const auto tie = [&](ValueRange inputs, ValueRange outputs) {
    if (countIfQubit(inputs) != countIfQubit(outputs)) {
      return;
    }
    for (auto [input, output] :
         llvm::zip_equal(quantumValues(inputs), quantumValues(outputs))) {
      origins[output] = origin(input);
    }
  };
  auto result = root->walk([&](Operation* op, const WalkStage& stage) {
    if (auto unitary = dyn_cast<qco::UnitaryOpInterface>(op)) {
      tie(unitary.getInputQubits(), unitary.getOutputQubits());
      return WalkResult::skip();
    }
    if (!stage.isAfterAllRegions()) {
      return WalkResult::advance();
    }
    bool positional = true;
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      positional = corresponds(loop.getRegionIterArgs(),
                               loop.getBody()->getTerminator()->getOperands());
      tie(loop.getInitArgs(), loop.getResults());
    } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      positional =
          corresponds(loop.getBeforeArguments(),
                      loop.getConditionOp().getArgs()) &&
          corresponds(loop.getAfterArguments(), loop.getYieldOp().getResults());
      tie(loop.getInits(), loop.getResults());
    } else if (isa<qco::IfOp, qco::IndexSwitchOp>(op)) {
      for (auto& region : op->getRegions()) {
        positional &=
            region.hasOneBlock() &&
            corresponds(region.front().getArguments(),
                        region.front().getTerminator()->getOperands());
      }
      tie(op->getOperands(), op->getResults());
    } else if (auto function = dyn_cast<func::FuncOp>(op)) {
      auto arguments = getQuantumArgumentIndices(function.getArgumentTypes());
      if (arguments.empty()) {
        return WalkResult::advance();
      }
      if (function.getNumResults() < arguments.size() ||
          !llvm::all_of(
              llvm::zip_equal(arguments, function.getResultTypes().take_back(
                                             arguments.size())),
              [&](auto pair) {
                auto [argument, type] = pair;
                return function.getArgumentTypes()[argument] == type;
              })) {
        function.emitOpError(
            "must return one trailing quantum value for each quantum argument");
        return WalkResult::interrupt();
      }
      if (function.isDeclaration()) {
        return WalkResult::advance();
      }
      if (!function.getBody().hasOneBlock()) {
        function.emitOpError(
            "with quantum arguments must have one outer block");
        return WalkResult::interrupt();
      }
      auto returned =
          dyn_cast<func::ReturnOp>(function.getBody().front().back());
      if (!returned ||
          !corresponds(function.getArguments(),
                       returned.getOperands().take_back(arguments.size()))) {
        function.emitOpError("must return its quantum arguments positionally");
        return WalkResult::interrupt();
      }
    } else if (auto measure = dyn_cast<qco::MeasureOp>(op)) {
      tie(ValueRange{measure.getQubitIn()}, ValueRange{measure.getQubitOut()});
    } else if (auto reset = dyn_cast<qco::ResetOp>(op)) {
      tie(ValueRange{reset.getQubitIn()}, ValueRange{reset.getQubitOut()});
    } else if (auto extract = dyn_cast<qtensor::ExtractOp>(op)) {
      origins[extract->getResult(0)] = origin(extract.getTensor());
    } else if (auto insert = dyn_cast<qtensor::InsertOp>(op)) {
      origins[insert.getResult()] = origin(insert.getDest());
    } else if (auto call = dyn_cast<func::CallOp>(op)) {
      for (auto [index, result] : llvm::enumerate(call.getResults())) {
        if (auto argument = qco::getCallArgumentForResult(call, index);
            succeeded(argument)) {
          origins[result] = origin(call.getOperand(*argument));
        }
      }
    }
    if (!positional) {
      op->emitOpError("requires positional quantum state "
                      "correspondence through region terminators");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

[[nodiscard]] static LogicalResult verifyLinearValue(Value value) {
  if (!isLinearQubitType(value.getType()) || value.hasOneUse()) {
    return success();
  }
  return emitError(value.getLoc())
         << "expected linear QCO value to have exactly one use, but found "
         << value.getNumUses();
}

LogicalResult mlir::qco::verifyLinearity(Operation* root) {
  func::FuncOp entryPoint;
  if (auto moduleOp = dyn_cast<ModuleOp>(root)) {
    entryPoint = mqt::getEntryPoint(moduleOp);
  }

  DenseSet<uint64_t> staticIndices;
  const auto walkResult = root->walk([&](Operation* op) {
    if (auto staticOp = dyn_cast<StaticOp>(op)) {
      if (entryPoint &&
          (entryPoint.isDeclaration() ||
           staticOp->getBlock() != &entryPoint.getBody().front())) {
        staticOp.emitError()
            << "expected static qubits in the entry block of program entry "
               "function @"
            << entryPoint.getSymName();
        return WalkResult::interrupt();
      }
      if (!staticIndices.insert(staticOp.getIndex()).second) {
        staticOp.emitError()
            << "expected each static qubit index to identify one linear "
               "value, but found duplicate index "
            << staticOp.getIndex();
        return WalkResult::interrupt();
      }
    }
    for (auto result : op->getResults()) {
      if (failed(verifyLinearValue(result))) {
        return WalkResult::interrupt();
      }
    }
    for (Region& region : op->getRegions()) {
      for (Block& block : region) {
        for (auto argument : block.getArguments()) {
          if (failed(verifyLinearValue(argument))) {
            return WalkResult::interrupt();
          }
        }
      }
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return failure();
  }
  DenseMap<Value, Value> origins;
  return collectWireOrigins(root, origins);
}
