/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Flux/FluxUtils.h"
#include "mlir/Dialect/Flux/IR/FluxDialect.h"

#include <cstddef>
#include <functional>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::flux;

size_t BarrierOp::getNumQubits() { return getNumTargets(); }

size_t BarrierOp::getNumTargets() { return getQubitsIn().size(); }

size_t BarrierOp::getNumControls() { return 0; }

size_t BarrierOp::getNumPosControls() { return 0; }

size_t BarrierOp::getNumNegControls() { return 0; }

Value BarrierOp::getInputQubit(const size_t i) { return getInputTarget(i); }

Value BarrierOp::getOutputQubit(const size_t i) { return getOutputTarget(i); }

Value BarrierOp::getInputTarget(const size_t i) {
  if (i < getNumTargets()) {
    return getQubitsIn()[i];
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

Value BarrierOp::getOutputTarget(const size_t i) {
  if (i < getNumTargets()) {
    return getQubitsOut()[i];
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

Value BarrierOp::getInputPosControl(const size_t i) {
  llvm::reportFatalUsageError("BarrierOp cannot be controlled");
}

Value BarrierOp::getOutputPosControl(const size_t i) {
  llvm::reportFatalUsageError("BarrierOp cannot be controlled");
}

Value BarrierOp::getInputNegControl(const size_t i) {
  llvm::reportFatalUsageError("BarrierOp cannot be controlled");
}

Value BarrierOp::getOutputNegControl(const size_t i) {
  llvm::reportFatalUsageError("BarrierOp cannot be controlled");
}

Value BarrierOp::getInputForOutput(const Value output) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (output == getQubitsOut()[i]) {
      return getQubitsIn()[i];
    }
  }
  llvm::report_fatal_error("Given qubit is not an output of the operation");
}

Value BarrierOp::getOutputForInput(const Value input) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (input == getQubitsIn()[i]) {
      return getQubitsOut()[i];
    }
  }
  llvm::report_fatal_error("Given qubit is not an input of the operation");
}

size_t BarrierOp::getNumParams() { return 0; }

Value BarrierOp::getParameter(const size_t i) {
  llvm::reportFatalUsageError("BarrierOp has no parameters");
}

void BarrierOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                      const ValueRange qubits) {
  SmallVector<Type> resultTypes;
  resultTypes.reserve(qubits.size());
  for (const Value qubit : qubits) {
    resultTypes.push_back(qubit.getType());
  }
  build(odsBuilder, odsState, resultTypes, qubits);
}
