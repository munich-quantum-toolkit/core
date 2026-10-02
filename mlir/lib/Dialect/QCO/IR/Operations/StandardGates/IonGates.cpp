/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Utils/Matrix.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"

#include <complex>
#include <numbers>
#include <optional>
#include <variant>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::mqt;

void GPIOp::build(OpBuilder& builder, OperationState& state, Value qubitIn,
                  const std::variant<double, Value>& phi) {
  Value phiValue = variantToValue(builder, state.location, phi);
  build(builder, state, qubitIn, phiValue);
}

Matrix2x2 GPIOp::unitaryMatrix(double phi) {
  return std::complex<double>{0., 1.} *
         ROp::unitaryMatrix(std::numbers::pi, phi);
}

std::optional<Matrix2x2> GPIOp::getUnitaryMatrix() {
  const auto phi = valueToDouble(getPhi());
  if (!phi) {
    return std::nullopt;
  }
  return unitaryMatrix(*phi);
}

void GPI2Op::build(OpBuilder& builder, OperationState& state, Value qubitIn,
                   const std::variant<double, Value>& phi) {
  Value phiValue = variantToValue(builder, state.location, phi);
  build(builder, state, qubitIn, phiValue);
}

Matrix2x2 GPI2Op::unitaryMatrix(double phi) {
  return ROp::unitaryMatrix(std::numbers::pi / 2., phi);
}

std::optional<Matrix2x2> GPI2Op::getUnitaryMatrix() {
  const auto phi = valueToDouble(getPhi());
  if (!phi) {
    return std::nullopt;
  }
  return unitaryMatrix(*phi);
}
