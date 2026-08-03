/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace mlir::qco {

static SinkOp findDeallocForAlloc(AllocOp alloc) {
  Value currentValue = alloc.getResult();
  uint64_t currentIndexInTensor = 0;
  bool isInTensor = false;

  while (currentValue) {
    // Both qubits and qubit tensors are linear values, so every step of the
    // chain has exactly one user.
    if (!currentValue.hasOneUse()) {
      return nullptr;
    }
    auto* user = *currentValue.getUsers().begin();

    if (isInTensor) {
      // The qubit currently lives at `currentIndexInTensor` of the tensor in
      // `currentValue`. Follow the tensor until it is extracted again.
      if (auto extractOp = dyn_cast<qtensor::ExtractOp>(user)) {
        const auto index = getConstantIntValue(extractOp.getIndex());
        if (!index) {
          // Dynamic index, cannot tell whether it is our qubit.
          return nullptr;
        }
        if (static_cast<uint64_t>(*index) == currentIndexInTensor) {
          currentValue = extractOp.getResult();
          isInTensor = false;
        } else {
          currentValue = extractOp.getOutTensor();
        }
        continue;
      }
      if (auto insertOp = dyn_cast<qtensor::InsertOp>(user)) {
        const auto index = getConstantIntValue(insertOp.getIndex());
        if (!index || static_cast<uint64_t>(*index) == currentIndexInTensor) {
          // Dynamic index, or our slot is overwritten by another qubit.
          return nullptr;
        }
        currentValue = insertOp.getResult();
        continue;
      }
      // Anything else (a dealloc, a call, ...) takes the qubit out of reach.
      return nullptr;
    }

    if (auto deallocOp = dyn_cast<SinkOp>(user)) {
      return deallocOp;
    }
    if (auto unitaryOp = dyn_cast<UnitaryOpInterface>(user)) {
      currentValue = unitaryOp.getOutputForInput(currentValue);
      continue;
    }
    if (auto measureOp = dyn_cast<MeasureOp>(user)) {
      currentValue = measureOp.getQubitOut();
      continue;
    }
    if (auto resetOp = dyn_cast<ResetOp>(user)) {
      currentValue = resetOp.getQubitOut();
      continue;
    }
    if (isa<func::CallOp>(user)) {
      // Relies on the QCO calling convention that the i-th qubit result of a
      // call corresponds to its i-th qubit operand.
      // TODO-Damian this only works if the indices are the same. Implement a
      // helper function to get the index
      for (auto i = 0ULL; i < user->getNumOperands(); i++) {
        if (user->getOperand(i) == currentValue) {
          currentValue = user->getResult(i);
          break;
        }
      }
      continue;
    }
    if (auto fromElementsOp = dyn_cast<qtensor::FromElementsOp>(user)) {
      for (auto i = 0ULL; i < user->getNumOperands(); i++) {
        if (user->getOperand(i) == currentValue) {
          currentIndexInTensor = i;
          isInTensor = true;
          break;
        }
      }
      currentValue = fromElementsOp.getResult();
      continue;
    }
    if (auto insertOp = dyn_cast<qtensor::InsertOp>(user)) {
      const auto index = getConstantIntValue(insertOp.getIndex());
      if (!index) {
        return nullptr;
      }
      currentIndexInTensor = static_cast<uint64_t>(*index);
      isInTensor = true;
      currentValue = insertOp.getResult();
      continue;
    }
    if (user->getNumResults() != 1) {
      // Multiple results, should not happen.
      return nullptr;
    }
    currentValue = user->getResult(0);
  }
  return nullptr;
}

static bool isRecursiveHelper(CallGraphNode* current, CallGraphNode* target,
                              llvm::DenseSet<CallGraphNode*>& visited) {
  if (!visited.insert(current).second) {
    return false; // Already visited
  }

  for (const auto& edge : *current) {
    CallGraphNode* callee = edge.getTarget();
    if (callee == target) {
      return true;
    }
    if (isRecursiveHelper(callee, target, visited)) {
      return true;
    }
  }

  return false;
}

static bool isRecursive(CallGraph& cg, func::FuncOp func) {
  CallGraphNode* node = cg.lookupNode(func.getCallableRegion());
  if (node == nullptr) {
    return false;
  }

  llvm::DenseSet<CallGraphNode*> visited;
  // Start from the function's callees to avoid immediately returning true
  for (const auto& edge : *node) {
    if (isRecursiveHelper(edge.getTarget(), node, visited)) {
      return true;
    }
  }

  return false;
}

static void tryAuxiliaryQubitHoisting(func::FuncOp funcOp) {
  funcOp.walk([&](AllocOp allocOp) {
    if (allocOp->getBlock()->getParentOp() != funcOp) {
      // Not directly in the function body, skip.
      return;
    }

    auto dealloc = findDeallocForAlloc(allocOp);

    if (!dealloc) {
      // No matching dealloc found, skip.
      return;
    }

    // Add a block argument for the auxiliary qubit.
    OpBuilder builder(dealloc);
    auto* block = allocOp->getBlock();
    auto loc = allocOp.getLoc();
    auto qubitType = allocOp.getType();
    auto newArg = block->addArgument(qubitType, loc);

    // Replace all uses of the alloc with the new block argument.
    allocOp.replaceAllUsesWith(newArg);

    // Erase the original alloc operation.
    allocOp.erase();

    // Replace the dealloc with a reset
    builder.setInsertionPoint(dealloc);
    auto resetOp =
        builder.create<ResetOp>(dealloc.getLoc(), dealloc.getQubit());
    dealloc.erase();

    // Add reset outcome to function results and alloc to function arguments
    auto funcType = funcOp.getFunctionType();
    SmallVector<Type> newArgTypes(funcType.getInputs().begin(),
                                  funcType.getInputs().end());
    SmallVector<Type> newResultTypes(funcType.getResults().begin(),
                                     funcType.getResults().end());
    newArgTypes.push_back(newArg.getType());
    newResultTypes.push_back(resetOp.getResult().getType());
    auto newFuncType =
        FunctionType::get(funcOp.getContext(), newArgTypes, newResultTypes);
    funcOp.setType(newFuncType);

    // Also add reset outcome to return
    funcOp.walk([&](func::ReturnOp returnOp) {
      OpBuilder returnBuilder(returnOp);
      SmallVector<Value> newReturnValues(returnOp.getOperands().begin(),
                                         returnOp.getOperands().end());
      newReturnValues.push_back(resetOp.getResult());
      returnBuilder.create<func::ReturnOp>(returnOp.getLoc(), newReturnValues);
      returnOp.erase();
    });

    // Update all call sites to handle the new return value
    // We use the SymbolTable to find all calls to this function
    if (auto uses = SymbolTable::getSymbolUses(funcOp, funcOp->getParentOp())) {
      for (auto use : *uses) {
        if (auto callOp = dyn_cast<func::CallOp>(use.getUser())) {
          builder.setInsertionPoint(callOp);

          // A. Add new alloc
          auto newAlloc = builder.create<AllocOp>(loc);

          // B. Create New Call
          SmallVector<Value> newCallOperands =
              llvm::to_vector(callOp.getOperands());
          newCallOperands.push_back(newAlloc);
          auto newCall =
              builder.create<func::CallOp>(loc, funcOp, newCallOperands);

          // C. Add dealloc after call
          builder.create<SinkOp>(
              loc, newCall.getResult(newCall.getNumResults() - 1));
          for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
            callOp.getResult(i).replaceAllUsesWith(newCall.getResult(i));
          }
          callOp.erase();
        }
      }
    }
  });
}

void runAuxiliaryQubitHoisting(ModuleOp module) {
  SmallVector<func::FuncOp> hoistingCandidates;
  CallGraph callGraph(module);

  module.walk([&](func::FuncOp func) {
    if (func.isPublic() || func.isDeclaration()) {
      return;
    }
    if (isRecursive(callGraph, func)) {
      return;
    }
    hoistingCandidates.push_back(func);
  });

  for (auto& func : hoistingCandidates) {
    tryAuxiliaryQubitHoisting(func);

    RewritePatternSet patterns(module.getContext());
    if (!applyPatternsGreedily(module, std::move(patterns)).succeeded()) {
      throw std::runtime_error("Failed to apply reuse qubits patterns after "
                               "auxiliary qubit hoisting.");
    }
  }
}

} // namespace mlir::qco
