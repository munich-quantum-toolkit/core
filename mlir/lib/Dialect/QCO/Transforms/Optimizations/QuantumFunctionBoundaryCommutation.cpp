/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "IPOUtils.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h" // IWYU pragma: keep (Passes.h.inc)

#include <llvm/ADT/StringMap.h>
#include <mlir/Dialect/Arith/IR/Arith.h> // IWYU pragma: keep (Passes.h.inc)
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <string>

namespace mlir::qco {

#define GEN_PASS_DEF_QUANTUMFUNCTIONBOUNDARYCOMMUTATION
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

/**
 * @brief Check if two single-qubit unitary operations cancel each other out
 * because they are self-inverse.
 *
 * @param first The first unitary operation.
 * @param second The second unitary operation.
 * @return true if the operations cancel each other out, false otherwise.
 */
static bool doOpsCancel(UnitaryOpInterface first, UnitaryOpInterface second) {
  if (first.getNumQubits() != 1) {
    return false;
  }
  if (first.getOperation()->getName() != second.getOperation()->getName()) {
    return false;
  }
  if (isa<XOp, YOp, ZOp, HOp>(first)) {
    return true;
  }
  return false;
}

/// Caches the specialization created for a callee, per parameter index. The
/// gate that is removed inside the callee belongs to one specific argument, so
/// specializations must not be shared between parameters.
using BoundarySpecializations =
    llvm::StringMap<DenseMap<uint32_t, func::FuncOp>>;

/**
 * @brief Cancel a gate in front of a call against the same gate at the start of
 * the callee.
 *
 * @details
 * When the operation producing the argument at @p parameter and the first
 * operation applied to that argument inside the callee are the same
 * self-inverse gate, both can be dropped. The caller-side gate is erased and
 * the call is redirected to a copy of the callee without the callee-side gate.
 * Copies are cached per callee and parameter, so repeated call sites share one
 * specialization while different parameters get their own.
 *
 * @param call The call to look at.
 * @param symbolTable The symbol table of the surrounding module.
 * @param parameter The index of the qubit argument to consider.
 * @param previousSpecializations Cache of already-created specializations.
 */
static void
tryBoundaryCommutation(func::CallOp call, SymbolTable& symbolTable,
                       uint32_t parameter,
                       BoundarySpecializations& previousSpecializations,
                       SmallVectorImpl<func::FuncOp>* touchedFunctions) {
  auto calleeName = call.getCallee();
  auto funcOp = symbolTable.lookup<func::FuncOp>(calleeName);

  if (!funcOp || funcOp.isExternal()) {
    return;
  }

  auto argOutside = call.getArgOperands()[parameter];
  auto argInside = funcOp.getArgument(parameter);

  if (!argInside.hasOneUse()) {
    return;
  }
  if (argOutside.getDefiningOp() == nullptr) {
    return;
  }

  auto lastOp = dyn_cast<UnitaryOpInterface>(argOutside.getDefiningOp());
  auto nextOp = dyn_cast<UnitaryOpInterface>(*argInside.getUsers().begin());

  if (!lastOp || !nextOp) {
    return;
  }

  if (!doOpsCancel(lastOp, nextOp)) {
    return;
  }
  argOutside.replaceAllUsesWith(lastOp.getInputQubit(0));
  lastOp.erase();

  // The call is about to be redirected away from `funcOp`, so it may lose its
  // last caller.
  if (touchedFunctions != nullptr) {
    touchedFunctions->emplace_back(funcOp);
  }

  if (const auto it = previousSpecializations.find(calleeName);
      it != previousSpecializations.end()) {
    if (const auto cached = it->second.find(parameter);
        cached != it->second.end()) {
      call.setCallee(cached->second.getName());
      return;
    }
  }

  auto newFunc = copyFunction(funcOp, funcOp.getName().str() +
                                          "_spec_boundary_commutation_arg_" +
                                          std::to_string(parameter));
  symbolTable.insert(newFunc);

  auto newParameter = newFunc.getArgument(parameter);
  auto newUser = dyn_cast<UnitaryOpInterface>(*newParameter.getUsers().begin());

  for (auto i = 0U; i < newUser.getNumQubits(); ++i) {
    newUser.getOutputQubit(i).replaceAllUsesWith(newUser.getInputQubit(i));
  }
  newUser.erase();
  previousSpecializations[calleeName][parameter] = newFunc;
  if (touchedFunctions != nullptr) {
    touchedFunctions->emplace_back(newFunc);
  }

  call.setCallee(newFunc.getName());
}

/**
 * @brief Cancel gates across every call boundary in the module.
 *
 * @param moduleOp The module to transform.
 * @param symbolTable The symbol table of @p moduleOp.
 */
namespace {
/// Cancels a self-inverse gate in front of a call against the same gate at the
/// start of the callee.
struct QuantumFunctionBoundaryCommutation final
    : impl::QuantumFunctionBoundaryCommutationBase<
          QuantumFunctionBoundaryCommutation> {
  using impl::QuantumFunctionBoundaryCommutationBase<
      QuantumFunctionBoundaryCommutation>::
      QuantumFunctionBoundaryCommutationBase;

protected:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    BoundarySpecializations previousSpecializations;
    // Callees this pass redirects calls away from, plus the copies it creates.
    SmallVector<func::FuncOp> touchedFunctions;

    // Collect the calls first: the commutation erases the caller-side gate,
    // which would invalidate a walk in progress.
    SmallVector<func::CallOp> calls;
    moduleOp.walk([&](func::CallOp call) { calls.emplace_back(call); });

    for (auto call : calls) {
      for (uint32_t i = 0; i < call.getArgOperands().size(); ++i) {
        const auto arg = call.getArgOperands()[i];
        if (!isa<QubitType>(arg.getType())) {
          continue;
        }
        tryBoundaryCommutation(call, symbolTable, i, previousSpecializations,
                               &touchedFunctions);
      }
    }

    // Drop the callees this pass left without callers.
    eraseOrphanedSpecializations(symbolTable, touchedFunctions);
  }
};
} // namespace

} // namespace mlir::qco
