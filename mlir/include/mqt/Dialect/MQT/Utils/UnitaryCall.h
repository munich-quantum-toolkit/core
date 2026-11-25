/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mqt/Dialect/MQT/IR/MQTDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::mqt {

/// Verify the signature and acyclicity of a QC or QCO unitary call.
template <typename CallOp>
[[nodiscard]] LogicalResult verifyUnitaryCall(CallOp call,
                                              SymbolTableCollection& symbols) {
  auto callee =
      symbols.lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
  if (!callee) {
    return call->emitOpError() << "'" << call.getCallee()
                               << "' does not reference a valid function";
  }
  if (!isUnitaryFunction(callee)) {
    return call->emitOpError() << "callee '" << call.getCallee()
                               << "' is not marked with mqt.unitary";
  }
  if (callee.getArgumentTypes() != call->getOperandTypes()) {
    return call->emitOpError() << "operand types " << call->getOperandTypes()
                               << " do not match callee argument types "
                               << callee.getArgumentTypes();
  }
  if (callee.getResultTypes() != call->getResultTypes()) {
    return call->emitOpError()
           << "result types " << call->getResultTypes()
           << " do not match callee result types " << callee.getResultTypes();
  }

  auto caller = call->template getParentOfType<func::FuncOp>();
  if (!isUnitaryFunction(caller)) {
    return success();
  }
  // Every cycle contains a self-call or a call to a later definition.
  // OpenQASM emits definitions before their callers.
  if (caller != callee && caller->getBlock() == callee->getBlock() &&
      callee->isBeforeInBlock(caller)) {
    return success();
  }

  SmallVector<func::FuncOp> worklist{callee};
  DenseSet<Operation*> visited{callee};
  while (!worklist.empty()) {
    auto current = worklist.pop_back_val();
    if (current == caller) {
      return caller.emitError("unitary function must not be recursive");
    }
    current.walk([&](CallOp nested) {
      auto target = symbols.lookupNearestSymbolFrom<func::FuncOp>(
          nested, nested.getCalleeAttr());
      if (target && isUnitaryFunction(target) &&
          visited.insert(target).second) {
        worklist.push_back(target);
      }
    });
  }
  return success();
}

} // namespace mlir::mqt
