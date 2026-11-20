/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QIR/Utils/QIRUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::qir {

LogicalResult moveTerminalMeasurements(Block& body, Block& measurements) {
  DenseSet<Value> measuredQubits;
  SmallVector<LLVM::CallOp> measurementCalls;
  for (auto& operation : body) {
    if (auto store = dyn_cast<LLVM::StoreOp>(operation);
        store && store->hasAttr(QIR_QUBIT_STORE_ATTR) &&
        measuredQubits.contains(store.getValue())) {
      return store.emitError(
          "QIR Base Profile forbids using a qubit after measurement");
    }
    auto call = dyn_cast<LLVM::CallOp>(operation);
    if (!call) {
      continue;
    }
    const bool isMeasurement = call.getCallee() == QIR_MEASURE;
    if (isMeasurement && call.getNumOperands() != 2) {
      return call.emitError("QIR measurement expects a qubit and a result");
    }
    // The second measurement operand is a result pointer.
    auto operands = call.getArgOperands();
    if (isMeasurement) {
      operands = operands.take_front(1);
    }
    for (auto operand : operands) {
      if (measuredQubits.contains(operand)) {
        return call.emitError(
            "QIR Base Profile forbids using a qubit after measurement");
      }
    }
    if (isMeasurement) {
      measuredQubits.insert(call.getOperand(0));
      measurementCalls.push_back(call);
    }
  }
  for (auto call : measurementCalls) {
    call->moveBefore(measurements.getTerminator());
  }
  return success();
}

static bool isBaseConstant(Value value) {
  auto* op = value.getDefiningOp();
  if (op == nullptr) {
    return false;
  }
  if (isa<LLVM::ConstantOp, LLVM::ZeroOp, LLVM::AddressOfOp>(op)) {
    return true;
  }
  return isa<LLVM::IntToPtrOp, LLVM::GEPOp>(op) &&
         llvm::all_of(op->getOperands(), isBaseConstant);
}

LogicalResult verifyBaseInstructions(LLVM::LLVMFuncOp function) {
  DenseSet<Value> initializedResults;
  const auto result = function.getBody().walk([&](Operation* op) {
    const auto constantOperands =
        llvm::all_of(op->getOperands(), isBaseConstant);
    if (isa<LLVM::ConstantOp, LLVM::ZeroOp, LLVM::AddressOfOp>(op) ||
        (isa<LLVM::IntToPtrOp, LLVM::GEPOp, LLVM::ReturnOp>(op) &&
         constantOperands) ||
        (isa<LLVM::BrOp>(op) && op->getNumOperands() == 0)) {
      return WalkResult::advance();
    }
    if (auto call = dyn_cast<LLVM::CallOp>(op); call && call.getCallee() &&
                                                call.getNumResults() == 0 &&
                                                constantOperands) {
      const auto name = *call.getCallee();
      if (name == QIR_MEASURE) {
        if (call.getNumOperands() != 2) {
          call.emitError("QIR measurement expects a qubit and a result");
          return WalkResult::interrupt();
        }
        initializedResults.insert(call.getOperand(1));
      } else if (name == QIR_RECORD_OUTPUT &&
                 (call.getNumOperands() != 2 ||
                  !initializedResults.contains(call.getOperand(0)))) {
        call.emitError("QIR Base Profile requires every output bit to be "
                       "measured; use Adaptive Profile for initialized "
                       "classical outputs");
        return WalkResult::interrupt();
      }
      if (name.starts_with("__quantum__qis__") || name == QIR_INITIALIZE ||
          name == "__quantum__rt__tuple_record_output" ||
          name == QIR_ARRAY_RECORD_OUTPUT || name == QIR_RECORD_OUTPUT) {
        return WalkResult::advance();
      }
    }
    op->emitError("operation requires capabilities beyond QIR Base Profile");
    return WalkResult::interrupt();
  });
  return failure(result.wasInterrupted());
}

} // namespace mlir::qir
