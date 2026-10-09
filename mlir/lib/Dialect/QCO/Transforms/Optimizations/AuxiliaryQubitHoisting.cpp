/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QCO/Utils/FunctionUtils.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h" // IWYU pragma: keep

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Arith/IR/Arith.h" // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::qco {

#define GEN_PASS_DEF_AUXILIARYQUBITHOISTING
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

/// Return whether @p func takes part in a call cycle.
///
/// The search starts at the function's callees rather than at the function
/// itself, so a non-recursive function is not reported merely because the
/// search begins at its own node. A worklist instead of recursion keeps deep
/// call chains from exhausting the stack.
static bool isRecursive(CallGraph& cg, func::FuncOp func) {
  CallGraphNode* node = cg.lookupNode(func.getCallableRegion());
  if (node == nullptr) {
    return false;
  }

  llvm::DenseSet<CallGraphNode*> visited;
  SmallVector<CallGraphNode*> worklist;
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

/// Turn every auxiliary qubit of @p funcOp into an argument.
///
/// An auxiliary qubit is one the function allocates in its entry block and
/// releases with `qco.sink` in that same block. Hoisting it lets the caller own
/// the allocation and reuse one qubit across several calls, and moves
/// allocations toward the entry point, where lowering needs them. The qubit
/// becomes a new last argument, and its release becomes a `qco.reset` handed
/// back as a new last result, so the function still returns its quantum
/// arguments in argument order. Each call site allocates the qubit right before
/// the call and releases it right after, in the call's own block.
static void hoistAuxiliaryQubits(func::FuncOp funcOp) {
  // Collect the allocations up front: the loop below erases operations, which
  // would invalidate a walk in progress.
  Block& entryBlock = funcOp.getBody().front();
  auto allocOps = llvm::to_vector(entryBlock.getOps<AllocOp>());

  for (auto allocOp : allocOps) {
    auto release =
        dyn_cast_or_null<SinkOp>(findReleaseInBlock(allocOp.getResult()));
    if (!release) {
      continue;
    }

    // Collect the call sites before touching the signature. Every reference
    // has to be a direct call; a symbol captured anywhere else has no operand
    // list to extend and would keep pointing at the old signature.
    const auto uses = SymbolTable::getSymbolUses(funcOp, funcOp->getParentOp());
    if (!uses) {
      continue;
    }
    SmallVector<func::CallOp> callOps;
    const auto onlyDirectCalls = llvm::all_of(*uses, [&](const auto& use) {
      auto callOp = dyn_cast<func::CallOp>(use.getUser());
      if (!callOp || callOp.getCallee() != funcOp.getName()) {
        return false;
      }
      callOps.emplace_back(callOp);
      return true;
    });
    if (!onlyDirectCalls) {
      continue;
    }

    // Take the qubit as a new last argument and hand it back, reset, as a new
    // last result.
    const auto loc = allocOp.getLoc();
    auto argument = entryBlock.addArgument(allocOp.getType(), loc);
    allocOp.replaceAllUsesWith(argument);
    allocOp.erase();
    OpBuilder builder(release);
    auto reset = ResetOp::create(builder, release.getLoc(), release.getQubit());
    release.erase();

    auto returnOp = cast<func::ReturnOp>(entryBlock.getTerminator());
    SmallVector<Value> returned(returnOp.getOperands());
    returned.emplace_back(reset.getResult());
    returnOp->setOperands(returned);
    funcOp.setType(FunctionType::get(funcOp.getContext(),
                                     entryBlock.getArgumentTypes(),
                                     ValueRange(returned).getTypes()));

    for (auto callOp : callOps) {
      builder.setInsertionPoint(callOp);
      auto qubit = AllocOp::create(builder, loc);
      SmallVector<Value> operands(callOp.getOperands());
      operands.emplace_back(qubit);
      auto newCall =
          func::CallOp::create(builder, callOp.getLoc(), funcOp, operands);
      SinkOp::create(builder, loc, newCall.getResults().back());
      callOp->replaceAllUsesWith(newCall.getResults().drop_back());
      callOp.erase();
    }
  }
}

/// Order the hoisting candidates so that callees come before callers.
///
/// Recursive functions are not candidates, so the graph is acyclic here. The
/// traversal starts at the external caller node, leaving out functions no
/// entry point reaches; those have no call sites to hoist into anyway.
static SmallVector<func::FuncOp>
orderCalleesFirst(const CallGraph& cg, ArrayRef<func::FuncOp> candidates) {
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
    CallGraph callGraph(moduleOp);

    // Externally visible functions and declarations keep their signature.
    // Recursive functions are skipped because their allocation would have to
    // be threaded through every level of the recursion.
    SmallVector<func::FuncOp> candidates;
    for (auto func : moduleOp.getOps<func::FuncOp>()) {
      if (!func.isPublic() && !func.isDeclaration() &&
          func.getBody().hasOneBlock() && !isRecursive(callGraph, func)) {
        candidates.emplace_back(func);
      }
    }

    // Hoisting out of a callee puts an allocation into each of its callers,
    // which may itself be hoistable. Visiting callees first lets such an
    // allocation travel all the way up in one run, whatever order the module
    // declares the functions in.
    for (auto func : orderCalleesFirst(callGraph, candidates)) {
      hoistAuxiliaryQubits(func);
    }
  }
};
} // namespace

} // namespace mlir::qco
