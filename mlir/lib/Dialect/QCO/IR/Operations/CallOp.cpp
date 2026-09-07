/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <iterator>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

/// Query local effects and follow calls without recursively querying
/// containers. Summaries live only for this query: rewrites may change a
/// callee's effects.
static bool hasEffectFreeCallees(CallOp root) {
  SymbolTableCollection symbols;
  SmallPtrSet<Operation*, 8> active;
  SmallPtrSet<Operation*, 8> completed;
  SmallVector<std::pair<Operation*, bool>> worklist{{root, false}};
  while (!worklist.empty()) {
    auto [operation, leaving] = worklist.pop_back_val();
    if (leaving) {
      active.erase(operation);
      completed.insert(operation);
      continue;
    }
    if (auto call = dyn_cast<CallOp>(operation)) {
      auto symbol = call->getAttrOfType<FlatSymbolRefAttr>("callee");
      auto callee =
          symbol ? symbols.lookupNearestSymbolFrom<func::FuncOp>(call, symbol)
                 : func::FuncOp{};
      if (!callee || !mqt::isUnitaryFunction(callee) ||
          !callee.getBody().hasOneBlock() ||
          callee.getArgumentTypes() != call.getOperandTypes() ||
          callee.getResultTypes() != call.getResultTypes()) {
        return false;
      }
      if (completed.contains(callee)) {
        continue;
      }
      if (!active.insert(callee).second) {
        return false;
      }
      worklist.emplace_back(callee, true);
      for (Operation& nested : callee.getBody().front()) {
        worklist.emplace_back(&nested, false);
      }
      continue;
    }
    const bool recursive =
        operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
    if (auto effects = dyn_cast<MemoryEffectOpInterface>(operation)) {
      if (!effects.hasNoEffect()) {
        return false;
      }
    } else if (!recursive) {
      return false;
    }
    if (recursive) {
      for (Region& region : operation->getRegions()) {
        for (Block& block : region) {
          for (Operation& nested : block) {
            worklist.emplace_back(&nested, false);
          }
        }
      }
    }
  }
  return true;
}

void CallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  if (!hasEffectFreeCallees(*this)) {
    effects.emplace_back(MemoryEffects::Write::get());
  }
}

void CallOp::build(OpBuilder&, OperationState& state, FlatSymbolRefAttr callee,
                   ValueRange operands) {
  state.addAttribute("callee", callee);
  state.addOperands(operands);
  for (Value operand : operands) {
    if (isa<QubitType>(operand.getType())) {
      state.addTypes(operand.getType());
    }
  }
}

size_t CallOp::getNumParams() {
  return getNumOperands() < getNumResults()
             ? 0
             : getNumOperands() - getNumResults();
}

OperandRange CallOp::getParameters() {
  return getOperands().take_front(getNumParams());
}

OperandRange CallOp::getInputQubits() {
  return getOperands().drop_front(getNumParams());
}

Value CallOp::getInputForOutput(Value output) {
  auto result = dyn_cast<OpResult>(output);
  auto inputs = getInputQubits();
  if (!result || result.getOwner() != getOperation() ||
      result.getResultNumber() >= inputs.size()) {
    return {};
  }
  return inputs[result.getResultNumber()];
}

Value CallOp::getOutputForInput(Value input) {
  const auto position = llvm::find(getInputQubits(), input);
  if (position == getInputQubits().end()) {
    return {};
  }
  return getOutputQubit(
      static_cast<size_t>(std::distance(getInputQubits().begin(), position)));
}

LogicalResult CallOp::verify() {
  if (getNumOperands() < getNumResults() ||
      llvm::any_of(
          getParameters(),
          [](Value value) { return isa<QubitType>(value.getType()); }) ||
      llvm::any_of(getInputQubits(), [](Value value) {
        return !isa<QubitType>(value.getType());
      })) {
    return emitOpError(
        "requires one trailing qubit operand for every qubit result");
  }
  return success();
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
  auto function =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getCalleeAttr());
  if (!function) {
    return emitOpError() << "'" << getCallee()
                         << "' does not reference a valid function";
  }
  if (!mqt::isUnitaryFunction(function)) {
    return emitOpError() << "callee '" << getCallee()
                         << "' is not marked with mqt.unitary";
  }
  if (function.getArgumentTypes() != getOperandTypes()) {
    return emitOpError() << "operand types " << getOperandTypes()
                         << " do not match callee argument types "
                         << function.getArgumentTypes();
  }
  if (function.getResultTypes() != getResultTypes()) {
    return emitOpError() << "result types " << getResultTypes()
                         << " do not match callee result types "
                         << function.getResultTypes();
  }
  return success();
}
