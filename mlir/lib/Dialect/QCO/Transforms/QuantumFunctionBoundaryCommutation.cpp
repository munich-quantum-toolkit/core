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

mlir::func::FuncOp copyFunction(mlir::func::FuncOp funcOp,
                                mlir::StringRef newName) {
  OpBuilder builder(funcOp);
  builder.setInsertionPointAfter(funcOp);

  auto newFunc = funcOp.clone();
  newFunc.setName(newName.str());

  return newFunc;
}

bool doOpsCancel(qco::UnitaryOpInterface first,
                 qco::UnitaryOpInterface second) {
  // For now, let's just consider self-inverses and single-qubit, non-controlled
  // gates.
  if (first.getOperation()->getName() != second.getOperation()->getName()) {
    return false;
  }
  if (first.getNumQubits() != 1 || second.getNumQubits() != 1) {
    return false;
  }
  if (isa<qco::XOp, qco::YOp, qco::ZOp, qco::HOp>(first)) {
    return true;
  }
  return false;
}

void tryBoundaryCommutation(func::CallOp call, SymbolTable& symbolTable,
                            uint32_t parameter) {
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

  auto lastOp = dyn_cast<qco::UnitaryOpInterface>(argOutside.getDefiningOp());
  auto nextOp =
      dyn_cast<qco::UnitaryOpInterface>(*argInside.getUsers().begin());

  if (!lastOp || !nextOp) {
    return;
  }

  if (!doOpsCancel(lastOp, nextOp)) {
    return;
  }
  argOutside.replaceAllUsesWith(lastOp.getInputQubit(0));
  lastOp.erase();

  auto newFunc = copyFunction(funcOp, funcOp.getName().str() +
                                          "_spec_boundary_commutation");
  symbolTable.insert(newFunc);

  auto newParameter = newFunc.getArgument(parameter);
  auto newUser =
      mlir::dyn_cast<qco::UnitaryOpInterface>(*newParameter.getUsers().begin());

  for (auto i = 0U; i < newUser.getNumQubits(); ++i) {
    newUser.getOutputQubit(i).replaceAllUsesWith(newUser.getInputQubit(i));
  }
  newUser.erase();

  call.setCallee(newFunc.getName());
}

}; // end anonymous namespace

namespace mlir::qco {

void runQuantumFunctionBoundaryCommutation(ModuleOp module,
                                           SymbolTable& symbolTable) {
  module.walk([&](func::CallOp call) {
    for (uint32_t i = 0; i < call.getArgOperands().size(); ++i) {
      const auto arg = call.getArgOperands()[i];
      if (!isa<qco::QubitType>(arg.getType())) {
        continue;
      }
      tryBoundaryCommutation(call, symbolTable, i);
    }
  });
}

} // namespace mlir::qco
