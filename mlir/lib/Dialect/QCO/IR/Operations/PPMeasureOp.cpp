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

#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

void PPMeasureOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                        ValueRange qubitsIn, ArrayRef<StringRef> pauliProduct) {
  SmallVector<Type> resultTypes;
  resultTypes.reserve(qubitsIn.size());
  for (auto qubit : qubitsIn) {
    resultTypes.push_back(qubit.getType());
  }
  auto i1Type = odsBuilder.getIntegerType(1, false);
  build(odsBuilder, odsState, resultTypes, i1Type, qubitsIn,
        odsBuilder.getStrArrayAttr(pauliProduct), nullptr, nullptr, nullptr);
}

LogicalResult PPMeasureOp::verify() {
  const auto registerName = getRegisterName();
  const auto registerSize = getRegisterSize();
  const auto registerIndex = getRegisterIndex();

  const auto hasSize = registerSize.has_value();
  const auto hasIndex = registerIndex.has_value();
  const auto hasName = registerName.has_value();

  if (hasName != hasSize || hasName != hasIndex) {
    return emitOpError("register attributes must all be present or all absent");
  }

  if (hasName) {
    if (*registerIndex >= *registerSize) {
      return emitOpError("register_index (")
             << *registerIndex << ") must be less than register_size ("
             << *registerSize << ")";
    }
  }

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
