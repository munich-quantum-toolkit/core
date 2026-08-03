/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace mlir::qco {

static func::FuncOp copyFunction(func::FuncOp funcOp, StringRef newName) {
  auto newFunc = funcOp.clone();
  newFunc.setName(newName.str());

  return newFunc;
}

static bool doOpsCancel(UnitaryOpInterface first, UnitaryOpInterface second) {
  // For now, let's just consider self-inverses and single-qubit, non-controlled
  // gates.
  if (first.getOperation()->getName() != second.getOperation()->getName()) {
    return false;
  }
  if (isa<XOp, YOp, ZOp, HOp>(first)) {
    return true;
  }
  return false;
}

static void tryBoundaryCommutation(
    func::CallOp call, SymbolTable& symbolTable, uint32_t parameter,
    std::unordered_map<std::string, func::FuncOp>& previousSpecializations) {
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

  if (previousSpecializations.contains(funcOp.getName().str())) {
    call.setCallee(previousSpecializations[funcOp.getName().str()].getName());
    return;
  }

  auto newFunc = copyFunction(funcOp, funcOp.getName().str() +
                                          "_spec_boundary_commutation");
  symbolTable.insert(newFunc);

  auto newParameter = newFunc.getArgument(parameter);
  auto newUser = dyn_cast<UnitaryOpInterface>(*newParameter.getUsers().begin());

  for (auto i = 0U; i < newUser.getNumQubits(); ++i) {
    newUser.getOutputQubit(i).replaceAllUsesWith(newUser.getInputQubit(i));
  }
  newUser.erase();
  previousSpecializations[funcOp.getName().str()] = newFunc;

  call.setCallee(newFunc.getName());
}

void runQuantumFunctionBoundaryCommutation(ModuleOp module,
                                           SymbolTable& symbolTable) {
  std::unordered_map<std::string, func::FuncOp> previousSpecializations;
  module.walk([&](func::CallOp call) {
    for (uint32_t i = 0; i < call.getArgOperands().size(); ++i) {
      const auto arg = call.getArgOperands()[i];
      if (!isa<QubitType>(arg.getType())) {
        continue;
      }
      tryBoundaryCommutation(call, symbolTable, i, previousSpecializations);
    }
  });
}

} // namespace mlir::qco
