/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/IR/QCDialect.h"

#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/OperationSupport.h>

using namespace mlir;
using namespace mlir::qc;

size_t BarrierOp::getNumQubits() { return getNumTargets(); }

size_t BarrierOp::getNumTargets() { return getQubits().size(); }

size_t BarrierOp::getNumControls() { return 0; }

Value BarrierOp::getQubit(const size_t i) { return getTarget(i); }

Value BarrierOp::getTarget(const size_t i) {
  if (i < getNumTargets()) {
    return getQubits()[i];
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

Value BarrierOp::getControl(const size_t /*i*/) {
  llvm::reportFatalUsageError("BarrierOp cannot be controlled");
}

size_t BarrierOp::getNumParams() { return 0; }

Value BarrierOp::getParameter(const size_t /*i*/) {
  llvm::reportFatalUsageError("BarrierOp does not have parameters");
}
