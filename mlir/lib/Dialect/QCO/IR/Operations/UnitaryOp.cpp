/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/Utils/DenseUnitary.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <complex>
#include <cstdint>

using namespace mlir;
using namespace mlir::qco;

void UnitaryOp::build(OpBuilder& /*builder*/, OperationState& state,
                      ValueRange qubits, ElementsAttr matrix) {
  state.addOperands(qubits);
  state.addTypes(qubits.getTypes());
  state.addAttribute(getMatrixAttrName(state.name), matrix);
}

LogicalResult UnitaryOp::verify() {
  if (getQubitsOut().size() != getQubitsIn().size()) {
    return emitOpError("must return one qubit for every input qubit");
  }
  return mqt::verifyDenseUnitaryMatrix(getOperation(), getMatrix(),
                                       getQubitsIn());
}

Value UnitaryOp::getInputForOutput(Value output) {
  for (auto [input, candidate] :
       llvm::zip_equal(getQubitsIn(), getQubitsOut())) {
    if (candidate == output) {
      return input;
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an output of UnitaryOp");
}

Value UnitaryOp::getOutputForInput(Value input) {
  for (auto [candidate, output] :
       llvm::zip_equal(getQubitsIn(), getQubitsOut())) {
    if (candidate == input) {
      return output;
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an input of UnitaryOp");
}

DynamicMatrix UnitaryOp::getUnitaryMatrix() {
  const auto matrix = cast<DenseElementsAttr>(getMatrix());
  const auto dimension = matrix.getType().getShape()[0];
  DynamicMatrix result(dimension);
  auto values = matrix.getValues<std::complex<double>>();
  auto iterator = values.begin();
  for (int64_t row = 0; row < dimension; ++row) {
    for (int64_t column = 0; column < dimension; ++column) {
      result(row, column) = *iterator;
      ++iterator;
    }
  }
  return result;
}

LogicalResult UnitaryOp::fold(FoldAdaptor /*adaptor*/,
                              SmallVectorImpl<OpFoldResult>& results) {
  if (!mqt::isExactIdentityMatrix(getMatrix())) {
    return failure();
  }
  llvm::append_range(results, getQubitsIn());
  return success();
}
