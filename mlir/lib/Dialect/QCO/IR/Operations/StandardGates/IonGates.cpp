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
         ROp::unitaryMatrix(std::numbers::pi, 2. * std::numbers::pi * phi);
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
  return ROp::unitaryMatrix(std::numbers::pi / 2., 2. * std::numbers::pi * phi);
}

std::optional<Matrix2x2> GPI2Op::getUnitaryMatrix() {
  const auto phi = valueToDouble(getPhi());
  if (!phi) {
    return std::nullopt;
  }
  return unitaryMatrix(*phi);
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

Matrix4x4 MSOp::unitaryMatrix(double phi0, double phi1, double theta) {
  const auto phase0 =
      RZOp::unitaryMatrix(2. * std::numbers::pi * phi0).embedInTwoQubit(0);
  const auto phase1 =
      RZOp::unitaryMatrix(2. * std::numbers::pi * phi1).embedInTwoQubit(1);
  const auto phase = phase0 * phase1;
  return phase * RXXOp::unitaryMatrix(2. * std::numbers::pi * theta) *
         phase.adjoint();
}

std::optional<Matrix4x4> MSOp::getUnitaryMatrix() {
  const auto phi0 = valueToDouble(getPhi0());
  const auto phi1 = valueToDouble(getPhi1());
  const auto theta = valueToDouble(getTheta());
  if (!phi0 || !phi1 || !theta) {
    return std::nullopt;
  }
  return unitaryMatrix(*phi0, *phi1, *theta);
}

void ZZOp::build(OpBuilder& builder, OperationState& state, Value qubit0In,
                 Value qubit1In, const std::variant<double, Value>& theta) {
  Value thetaValue = variantToValue(builder, state.location, theta);
  build(builder, state, qubit0In, qubit1In, thetaValue);
}

Matrix4x4 ZZOp::unitaryMatrix(double theta) {
  return RZZOp::unitaryMatrix(2. * std::numbers::pi * theta);
}

std::optional<Matrix4x4> ZZOp::getUnitaryMatrix() {
  const auto theta = valueToDouble(getTheta());
  if (!theta) {
    return std::nullopt;
  }
  return unitaryMatrix(*theta);
}
