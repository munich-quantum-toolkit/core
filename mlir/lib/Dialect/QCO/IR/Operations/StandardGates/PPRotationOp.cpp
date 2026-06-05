/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>

using namespace mlir;
using namespace mlir::qco;

Value PPRotationOp::getInputTarget(const size_t i) {
  if (i < getNumTargets()) {
    return getQubitsIn()[i];
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

Value PPRotationOp::getOutputTarget(const size_t i) {
  if (i < getNumTargets()) {
    return getQubitsOut()[i];
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

Value PPRotationOp::getInputForOutput(Value output) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (output == getQubitsOut()[i]) {
      return getQubitsIn()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an output of the operation");
}

Value PPRotationOp::getOutputForInput(Value input) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (input == getQubitsIn()[i]) {
      return getQubitsOut()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an input of the operation");
}

bool PPRotationOp::isNonClifford() {
  auto piFraction = getRotation();
  return piFraction == 4 || piFraction == -4;
}

bool PPRotationOp::isClifford() { return !isNonClifford(); }

void PPRotationOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                         ValueRange qubitsIn, std::int8_t rotation,
                         ArrayRef<StringRef> pauliProduct) {
  SmallVector<Type> resultTypes;
  resultTypes.reserve(qubitsIn.size());
  for (auto qubit : qubitsIn) {
    resultTypes.push_back(qubit.getType());
  }
  auto si8Type = odsBuilder.getIntegerType(8, true);
  build(odsBuilder, odsState, resultTypes, qubitsIn,
        odsBuilder.getIntegerAttr(si8Type, rotation),
        odsBuilder.getStrArrayAttr(pauliProduct));
}

LogicalResult PPRotationOp::verify() {
  size_t numPaulis = getPauliProduct().size();
  if (numPaulis == 0) {
    return emitOpError("pauli_product must be non-empty");
  }
  if (numPaulis != getQubitsIn().size()) {
    return emitOpError("number of elements in pauli_product must match "
                       "number of input qubits");
  }
  for (const auto& pauli : getPauliProduct()) {
    const auto pauliStr = cast<StringAttr>(pauli).getValue();
    if (pauliStr != "I" && pauliStr != "X" && pauliStr != "Y" &&
        pauliStr != "Z") {
      return emitOpError("pauli_product elements must be one of 'I', 'X', "
                         "'Y', or 'Z'");
    }
  }
  return success();
}
