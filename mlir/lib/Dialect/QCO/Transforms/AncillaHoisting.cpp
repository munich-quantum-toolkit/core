/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

//
// Created by damian on 2/10/26.
//
#include "mlir/Analysis/CallGraph.h" // <--- The specialization is defined here
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include "llvm/ADT/SCCIterator.h" // <--- The iterator is defined here

#include <array>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {
using namespace mlir;

mlir::qco::DeallocOp findDeallocForAlloc(mlir::qco::AllocOp alloc) {
  Value currentValue = alloc.getResult();
  uint32_t currentIndexInTensor = 0;
  bool isInTensor = false;
  while (currentValue) {
    if (!currentValue.hasOneUse()) {
      if (isInTensor) {
        for (auto* user : currentValue.getUsers()) {
          if (auto extractOp = dyn_cast<mlir::tensor::ExtractOp>(user)) {
            const auto operands = extractOp.getIndices();
            if (operands.size() != 1) {
              // Not a 1D tensor, should not happen.
              return nullptr;
            }
            const auto indexValue = operands[0];
            if (auto constIndex = dyn_cast<arith::ConstantIndexOp>(
                    indexValue.getDefiningOp())) {
              if (constIndex.value() == currentIndexInTensor) {
                // Index does not match, should not happen.
                currentValue = extractOp.getResult();
                isInTensor = false;
                break;
              }
            }
          } else {
            return nullptr;
          }
        }
        if (isInTensor) {
          // No extract found, should not happen.
          return nullptr;
        }
        continue;
      } else {
        // Multiple users, should not happen.
        return nullptr;
      }
    }
    auto* user = *currentValue.getUsers().begin();
    if (auto deallocOp = dyn_cast<mlir::qco::DeallocOp>(user)) {
      return deallocOp;
    }
    if (auto unitaryOp = dyn_cast<mlir::qco::UnitaryOpInterface>(user)) {
      currentValue = unitaryOp.getOutputForInput(currentValue);
      continue;
    }
    if (auto measureOp = dyn_cast<mlir::qco::MeasureOp>(user)) {
      currentValue = measureOp.getQubitOut();
      continue;
    }
    if (auto resetOp = dyn_cast<mlir::qco::ResetOp>(user)) {
      currentValue = resetOp.getQubitOut();
      continue;
    }
    if (auto callOp = dyn_cast<mlir::func::CallOp>(user)) {
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
    if (auto fromElementsOp = dyn_cast<mlir::tensor::FromElementsOp>(user)) {
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
    if (user->getNumResults() != 1) {
      // Multiple results, should not happen.
      return nullptr;
    }
    currentValue = user->getResult(0);
  }
  return nullptr;
}

bool isRecursiveHelper(CallGraphNode* current, CallGraphNode* target,
                       llvm::DenseSet<CallGraphNode*>& visited) {
  if (!visited.insert(current).second)
    return false; // Already visited

  for (auto& edge : *current) {
    CallGraphNode* callee = edge.getTarget();
    if (callee == target)
      return true;
    if (isRecursiveHelper(callee, target, visited))
      return true;
  }

  return false;
}

bool isRecursive(CallGraph& cg, func::FuncOp func) {
  CallGraphNode* node = cg.lookupNode(func.getCallableRegion());
  if (!node)
    return false;

  llvm::DenseSet<CallGraphNode*> visited;
  // Start from the function's callees to avoid immediately returning true
  for (auto& edge : *node) {
    if (isRecursiveHelper(edge.getTarget(), node, visited))
      return true;
  }

  return false;
}

void tryAncillaHoisting(func::FuncOp funcOp, SymbolTable& symbolTable) {
  funcOp.walk([&](mlir::qco::AllocOp allocOp) {
    if (allocOp->getBlock()->getParentOp() != funcOp) {
      // Not directly in the function body, skip.
      return;
    }

    auto dealloc = findDeallocForAlloc(allocOp);

    if (!dealloc) {
      // No matching dealloc found, skip.
      return;
    }

    // Add a block argument for the ancilla qubit.
    OpBuilder builder(dealloc);
    auto block = allocOp->getBlock();
    auto loc = allocOp.getLoc();
    auto qubitType = allocOp.getType();
    auto newArg = block->addArgument(qubitType, loc);

    // Replace all uses of the alloc with the new block argument.
    allocOp.replaceAllUsesWith(newArg);

    // Erase the original alloc operation.
    allocOp.erase();

    // Replace the dealloc with a reset
    builder.setInsertionPoint(dealloc);
    auto resetOp = builder.create<mlir::qco::ResetOp>(dealloc.getLoc(),
                                                      dealloc.getQubit());
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
    if (auto uses = symbolTable.getSymbolUses(funcOp, funcOp->getParentOp())) {
      for (auto use : *uses) {
        if (auto callOp = dyn_cast<func::CallOp>(use.getUser())) {
          builder.setInsertionPoint(callOp);

          // A. Add new alloc
          auto newAlloc = builder.create<mlir::qco::AllocOp>(loc);

          // B. Create New Call
          SmallVector<Value> newCallOperands =
              llvm::to_vector(callOp.getOperands());
          newCallOperands.push_back(newAlloc);
          auto newCall =
              builder.create<func::CallOp>(loc, funcOp, newCallOperands);

          // C. Add dealloc after call
          builder.create<mlir::qco::DeallocOp>(
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

}; // end anonymous namespace

namespace mlir::qco {

void runAncillaHoisting(ModuleOp module, SymbolTable& symbolTable) {
  SmallVector<func::FuncOp> hoistingCandidates;
  mlir::CallGraph callGraph(module);

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
    tryAncillaHoisting(func, symbolTable);

    RewritePatternSet patterns(module.getContext());
    populateReuseQubitsPatterns(patterns);
    if (!applyPatternsGreedily(module, std::move(patterns)).succeeded()) {
      throw std::runtime_error(
          "Failed to apply reuse qubits patterns after ancilla hoisting.");
    }
  }
}

} // namespace mlir::qco
