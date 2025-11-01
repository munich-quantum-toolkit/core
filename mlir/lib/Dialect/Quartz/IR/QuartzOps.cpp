/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Quartz/IR/QuartzDialect.h" // IWYU pragma: associated
#include "mlir/Dialect/Utils/MatrixUtils.h"
#include "mlir/Dialect/Utils/ParameterDescriptor.h"

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
// IWYU pragma: end_keep

#include <cstddef>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::quartz;
using namespace mlir::utils;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quartz/IR/QuartzOpsDialect.cpp.inc"

void QuartzDialect::initialize() {
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Quartz/IR/QuartzOpsTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Quartz/IR/QuartzOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Quartz/IR/QuartzOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quartz/IR/QuartzInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Quartz/IR/QuartzOps.cpp.inc"

LogicalResult AllocOp::verify() {
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
  return success();
}

LogicalResult MeasureOp::verify() {
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
  return success();
}

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

// XOp

DenseElementsAttr XOp::tryGetStaticMatrix() {
  auto* ctx = getContext();
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  return DenseElementsAttr::get(type, getMatrixX());
}

// RXOp

ParameterDescriptor RXOp::getParameter(size_t i) {
  if (i == 0) {
    return {getThetaAttr(), getThetaDyn()};
  }
  llvm_unreachable("RXOp has one parameter");
}

bool RXOp::hasStaticUnitary() { return getParameter(0).isStatic(); }

DenseElementsAttr RXOp::tryGetStaticMatrix() {
  if (!hasStaticUnitary()) {
    return nullptr;
  }
  auto* ctx = getContext();
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const auto& theta = getTheta().value().convertToDouble();
  return DenseElementsAttr::get(type, getMatrixRX(theta));
}

LogicalResult RXOp::verify() {
  if (getTheta().has_value() == (getThetaDyn() != nullptr)) {
    return emitOpError("must specify exactly one of static or dynamic theta");
  }
  return success();
}

// U2Op

ParameterDescriptor U2Op::getParameter(size_t i) {
  if (i == 0) {
    return {getPhiAttr(), getPhiDyn()};
  }
  if (i == 1) {
    return {getLambdaAttr(), getLambdaDyn()};
  }
  llvm_unreachable("U2Op has two parameters");
}

bool U2Op::hasStaticUnitary() {
  return (getParameter(0).isStatic() && getParameter(1).isStatic());
}

DenseElementsAttr U2Op::tryGetStaticMatrix() {
  if (!hasStaticUnitary()) {
    return nullptr;
  }
  auto* ctx = getContext();
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const auto& phi = getPhi().value().convertToDouble();
  const auto& lambda = getLambda().value().convertToDouble();
  return DenseElementsAttr::get(type, getMatrixU2(phi, lambda));
}

LogicalResult U2Op::verify() {
  if (getPhi().has_value() == (getPhiDyn() != nullptr)) {
    return emitOpError("must specify exactly one of static or dynamic phi");
  }
  if (getLambda().has_value() == (getLambdaDyn() != nullptr)) {
    return emitOpError("must specify exactly one of static or dynamic lambda");
  }
  return success();
}

// SWAPOp

DenseElementsAttr SWAPOp::tryGetStaticMatrix() {
  auto* ctx = getContext();
  const auto& type =
      RankedTensorType::get({4, 4}, ComplexType::get(Float64Type::get(ctx)));
  return DenseElementsAttr::get(type, getMatrixSWAP());
}

//===----------------------------------------------------------------------===//
// Modifiers
//===----------------------------------------------------------------------===//

UnitaryOpInterface CtrlOp::getBodyUnitary() {
  return llvm::dyn_cast<UnitaryOpInterface>(&getBody().front().front());
}

size_t CtrlOp::getNumQubits() { return getNumTargets() + getNumControls(); }

size_t CtrlOp::getNumTargets() {
  auto unitaryOp = getBodyUnitary();
  return unitaryOp.getNumTargets();
}

size_t CtrlOp::getNumControls() {
  return getNumPosControls() + getNumNegControls();
}

size_t CtrlOp::getNumPosControls() { return getControls().size(); }

size_t CtrlOp::getNumNegControls() {
  auto unitaryOp = getBodyUnitary();
  return unitaryOp.getNumNegControls();
}

Value CtrlOp::getQubit(size_t i) {
  if (i < getNumTargets()) {
    return getTarget(i);
  }
  if (getNumTargets() <= i && i < getNumPosControls()) {
    return getPosControl(i - getNumTargets());
  }
  if (getNumTargets() + getNumPosControls() <= i && i < getNumQubits()) {
    return getNegControl(i - getNumTargets() - getNumPosControls());
  }
  llvm_unreachable("Invalid qubit index");
}

Value CtrlOp::getTarget(size_t i) {
  auto unitaryOp = getBodyUnitary();
  return unitaryOp.getTarget(i);
}

Value CtrlOp::getPosControl(size_t i) { return getControls()[i]; }

Value CtrlOp::getNegControl(size_t i) {
  auto unitaryOp = getBodyUnitary();
  return unitaryOp.getNegControl(i);
}

size_t CtrlOp::getNumParams() {
  auto unitaryOp = getBodyUnitary();
  return unitaryOp.getNumParams();
}

bool CtrlOp::hasStaticUnitary() {
  auto unitaryOp = getBodyUnitary();
  return unitaryOp.hasStaticUnitary();
}

ParameterDescriptor CtrlOp::getParameter(size_t i) {
  auto unitaryOp = getBodyUnitary();
  return unitaryOp.getParameter(i);
}

DenseElementsAttr CtrlOp::tryGetStaticMatrix() {
  llvm_unreachable("Not implemented yet"); // TODO
}
