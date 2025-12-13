/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"

#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/OperationSupport.h>

using namespace mlir;
using namespace mlir::quartz;

size_t BarrierOp::getNumQubits() { return getNumTargets(); }

size_t BarrierOp::getNumTargets() { return getQubits().size(); }

size_t BarrierOp::getNumControls() { return 0; }

size_t BarrierOp::getNumPosControls() { return 0; }

size_t BarrierOp::getNumNegControls() { return 0; }

Value BarrierOp::getQubit(const size_t i) { return getTarget(i); }

Value BarrierOp::getTarget(const size_t i) {
  if (i < getNumTargets()) {
    return getQubits()[i];
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

Value BarrierOp::getPosControl(const size_t /*i*/) {
  llvm::reportFatalUsageError("BarrierOp cannot be controlled");
}

Value BarrierOp::getNegControl(const size_t /*i*/) {
  llvm::reportFatalUsageError("BarrierOp cannot be controlled");
}

size_t BarrierOp::getNumParams() { return 0; }

Value BarrierOp::getParameter(const size_t /*i*/) {
  llvm::reportFatalUsageError("BarrierOp does not have parameters");
}
