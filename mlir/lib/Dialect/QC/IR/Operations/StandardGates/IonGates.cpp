/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Utils/Parameters.h"
#include "mqt/Dialect/QC/IR/QCOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"

#include <variant>

using namespace mlir;
using namespace mlir::qc;
using namespace mlir::mqt;

void GPIOp::build(OpBuilder& builder, OperationState& state, Value qubitIn,
                  const std::variant<double, Value>& phi) {
  Value phiValue = variantToValue(builder, state.location, phi);
  build(builder, state, qubitIn, phiValue);
}

void GPI2Op::build(OpBuilder& builder, OperationState& state, Value qubitIn,
                   const std::variant<double, Value>& phi) {
  Value phiValue = variantToValue(builder, state.location, phi);
  build(builder, state, qubitIn, phiValue);
}

void MSOp::build(OpBuilder& builder, OperationState& state, Value qubit0In,
                 Value qubit1In, const std::variant<double, Value>& phi0,
                 const std::variant<double, Value>& phi1,
                 const std::variant<double, Value>& theta) {
  Value phi0Value = variantToValue(builder, state.location, phi0);
  Value phi1Value = variantToValue(builder, state.location, phi1);
  Value thetaValue = variantToValue(builder, state.location, theta);
  build(builder, state, qubit0In, qubit1In, phi0Value, phi1Value, thetaValue);
}

void ZZOp::build(OpBuilder& builder, OperationState& state, Value qubit0In,
                 Value qubit1In, const std::variant<double, Value>& theta) {
  Value thetaValue = variantToValue(builder, state.location, theta);
  build(builder, state, qubit0In, qubit1In, thetaValue);
}
