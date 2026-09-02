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
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h" // IWYU pragma: keep (Passes.h.inc)
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h> // IWYU pragma: keep (Passes.h.inc)
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_AUXILIARYQUBITHOISTING
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

/**
 * @brief Find the operation that releases the qubit produced by @p alloc.
 *
 * @details
 * Follows the qubit forward along its linear use chain, through gates,
 * measurements, resets and calls, and also while it is parked inside a qubit
 * tensor. Returns a null op when the qubit escapes the function or when the
 * chain cannot be followed, for example because a tensor index is not known at
 * compile time.
 *
 * @param alloc The allocation whose release point is searched.
 * @param callMapping Resolves how qubits flow across call boundaries.
 * @return The `qco.sink` releasing the qubit, or a null op if there is none.
 */
static SinkOp findDeallocForAlloc(AllocOp alloc,
                                  CallQubitMapping& callMapping) {
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
        if (std::cmp_equal(*index, currentIndexInTensor)) {
          currentValue = extractOp.getResult();
          isInTensor = false;
        } else {
          currentValue = extractOp.getOutTensor();
        }
        continue;
      }
      if (auto insertOp = dyn_cast<qtensor::InsertOp>(user)) {
        const auto index = getConstantIntValue(insertOp.getIndex());
        if (!index || std::cmp_equal(*index, currentIndexInTensor)) {
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
    if (auto callOp = dyn_cast<func::CallOp>(user)) {
      const auto threadedOr =
          callMapping.getResultForOperand(callOp, currentValue);
      if (failed(threadedOr)) {
        return nullptr;
      }
      const auto threaded = *threadedOr;
      if (!threaded) {
        // The callee keeps the qubit, so it is never released here.
        return nullptr;
      }
      currentValue = threaded;
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
    // Anything else is not known to thread the qubit. Guessing that a single
    // result carries it on would silently follow an unrelated value, so give up
    // instead.
    return nullptr;
  }
  return nullptr;
}

/**
 * @brief Check whether the given function takes part in a call cycle.
 *
 * @details
 * Performs an iterative reachability search over the call graph, starting at
 * the function's callees rather than at the function itself, so that a
 * non-recursive function is not reported as recursive merely because the
 * search begins at its own node. A worklist is used instead of recursion so
 * that deep call chains cannot exhaust the stack.
 *
 * @param cg The call graph of the surrounding module.
 * @param func The function to check.
 * @return True if @p func can reach itself through a chain of calls.
 */
static bool isRecursive(CallGraph& cg, func::FuncOp func) {
  CallGraphNode* node = cg.lookupNode(func.getCallableRegion());
  if (node == nullptr) {
    return false;
  }

  llvm::DenseSet<CallGraphNode*> visited;
  SmallVector<CallGraphNode*> worklist;

  // Seed the search with the function's callees so that the function itself is
  // only reported as recursive when a call chain leads back to it.
  for (const auto& edge : *node) {
    worklist.emplace_back(edge.getTarget());
  }

  while (!worklist.empty()) {
    auto* current = worklist.pop_back_val();
    if (current == node) {
      return true;
    }
    if (!visited.insert(current).second) {
      continue;
    }
    for (const auto& edge : *current) {
      worklist.emplace_back(edge.getTarget());
    }
  }

  return false;
}

/**
 * @brief Turn every auxiliary qubit of the given function into an argument.
 *
 * @details
 * An auxiliary qubit is one that the function allocates and releases itself.
 * Hoisting it makes the caller own the allocation, which lets the caller reuse
 * one qubit across several calls. The release point becomes a `qco.reset` that
 * is handed back as an additional result, so the caller receives the qubit in a
 * known state.
 *
 * @param funcOp The function to transform.
 * @param callMapping Resolves how qubits flow across call boundaries.
 */
static void tryAuxiliaryQubitHoisting(func::FuncOp funcOp,
                                      CallQubitMapping& callMapping) {
  // The release point is rewritten into a reset whose result is appended to
  // every return. That is only sound while there is a single block, because a
  // reset in one block need not reach a return in another.
  if (!funcOp.getBody().hasOneBlock()) {
    return;
  }

  // Collect the allocations up front: the loop below erases operations, which
  // would invalidate a walk in progress.
  SmallVector<AllocOp> allocOps;
  funcOp.walk([&](AllocOp allocOp) {
    if (allocOp->getBlock()->getParentOp() != funcOp) {
      // Not directly in the function body, skip.
      return;
    }
    allocOps.emplace_back(allocOp);
  });

  for (auto allocOp : allocOps) {
    auto dealloc = findDeallocForAlloc(allocOp, callMapping);

    if (!dealloc) {
      // No matching dealloc found, skip.
      continue;
    }

    // Collect the call sites before touching the signature. Once the signature
    // changes the existing calls no longer match it, so if the uses cannot be
    // determined it must not have been changed in the first place.
    const auto uses = SymbolTable::getSymbolUses(funcOp, funcOp->getParentOp());
    if (!uses) {
      continue;
    }

    // Every reference has to be a direct call. Anything else, such as the
    // symbol captured in an attribute or taken as a function value, has no
    // operand list to extend and would be left pointing at the old signature.
    SmallVector<func::CallOp> callOps;
    auto onlyDirectCalls = true;
    for (const auto use : *uses) {
      auto callOp = dyn_cast<func::CallOp>(use.getUser());
      if (!callOp || callOp.getCallee() != funcOp.getName()) {
        onlyDirectCalls = false;
        break;
      }
      callOps.emplace_back(callOp);
    }
    if (!onlyDirectCalls) {
      continue;
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
        ResetOp::create(builder, dealloc.getLoc(), dealloc.getQubit());
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
    // The cached mapping describes the old signature, so it is stale now.
    callMapping.invalidate();

    // Also add the reset outcome to every return. The operands are updated in
    // place so that the terminator stays valid for the ongoing walk.
    funcOp.walk([&](func::ReturnOp returnOp) {
      SmallVector<Value> newReturnValues(returnOp.getOperands().begin(),
                                         returnOp.getOperands().end());
      newReturnValues.emplace_back(resetOp.getResult());
      returnOp->setOperands(newReturnValues);
    });

    // Update the call sites collected above to handle the new return value.
    for (auto callOp : callOps) {
      builder.setInsertionPoint(callOp);

      // A. Add new alloc
      auto newAlloc = AllocOp::create(builder, loc);

      // B. Create New Call
      SmallVector<Value> newCallOperands =
          llvm::to_vector(callOp.getOperands());
      newCallOperands.emplace_back(newAlloc);
      auto newCall =
          func::CallOp::create(builder, loc, funcOp, newCallOperands);

      // C. Add dealloc after call
      SinkOp::create(builder, loc,
                     newCall.getResult(newCall.getNumResults() - 1));
      for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
        callOp.getResult(i).replaceAllUsesWith(newCall.getResult(i));
      }
      callOp.erase();
    }
  }
}

/**
 * @brief Hoist the auxiliary qubits of every eligible function in the module.
 *
 * @details
 * Externally visible functions and declarations are skipped because their
 * signature cannot be changed, and recursive functions are skipped because
 * their allocation would have to be threaded through every level of the
 * recursion.
 *
 * @param moduleOp The module to transform.
 */
/**
 * @brief Order the hoisting candidates so that callees come before callers.
 *
 * @details
 * Recursive functions are not candidates, so the graph is acyclic here. The
 * traversal starts at the external caller node, leaving out functions no entry
 * point reaches; those have no call sites to hoist into anyway.
 *
 * @param cg The call graph of the surrounding module.
 * @param candidates The functions to order.
 * @return The candidates, callees first.
 */
static SmallVector<func::FuncOp>
orderCalleesFirst(const CallGraph& cg,
                  const SmallVector<func::FuncOp>& candidates) {
  llvm::DenseMap<CallGraphNode*, func::FuncOp> candidateNodes;
  for (auto func : candidates) {
    if (auto* node = cg.lookupNode(func.getCallableRegion())) {
      candidateNodes.try_emplace(node, func);
    }
  }

  SmallVector<func::FuncOp> ordered;
  ordered.reserve(candidates.size());
  for (auto* node : llvm::post_order(&cg)) {
    if (const auto it = candidateNodes.find(node); it != candidateNodes.end()) {
      ordered.emplace_back(it->second);
    }
  }
  return ordered;
}

namespace {
/// Turns qubits a callee allocates and releases itself into arguments.
struct AuxiliaryQubitHoisting final
    : impl::AuxiliaryQubitHoistingBase<AuxiliaryQubitHoisting> {
  using impl::AuxiliaryQubitHoistingBase<
      AuxiliaryQubitHoisting>::AuxiliaryQubitHoistingBase;

protected:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    SmallVector<func::FuncOp> hoistingCandidates;
    CallGraph callGraph(moduleOp);
    // One shared mapping so that each callee is threaded at most once.
    CallQubitMapping callMapping;

    moduleOp.walk([&](func::FuncOp func) {
      if (func.isPublic() || func.isDeclaration()) {
        return;
      }
      if (isRecursive(callGraph, func)) {
        return;
      }
      hoistingCandidates.push_back(func);
    });

    // Hoisting out of a callee puts an allocation into each of its callers,
    // which may itself be hoistable. Visiting callees first lets such an
    // allocation travel all the way up in a single run instead of stopping
    // wherever the module happens to declare the functions.
    for (auto func : orderCalleesFirst(callGraph, hoistingCandidates)) {
      tryAuxiliaryQubitHoisting(func, callMapping);
    }
  }
};
} // namespace

} // namespace mlir::qco
