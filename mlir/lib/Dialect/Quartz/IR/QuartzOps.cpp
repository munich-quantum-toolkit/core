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

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
// IWYU pragma: end_keep

#include <cmath>
#include <complex>
#include <cstddef>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::quartz;

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

Value XOp::getQubit(size_t i) {
  if (i != 0) {
    llvm_unreachable("XOp has only one qubit");
  }
  return getQubitIn();
}

Value XOp::getTarget(size_t i) {
  if (i != 0) {
    llvm_unreachable("XOp has only one target qubit");
  }
  return getQubitIn();
}

Value XOp::getPosControl(size_t /*i*/) {
  llvm_unreachable("XOp does not have controls");
}

Value XOp::getNegControl(size_t /*i*/) {
  llvm_unreachable("XOp does not have controls");
}

ParameterDescriptor XOp::getParameter(size_t /*i*/) {
  llvm_unreachable("XOp does not have parameters");
}

DenseElementsAttr XOp::tryGetStaticMatrix() {
  auto* ctx = getContext();
  auto type = RankedTensorType::get({2, 2}, Float64Type::get(ctx));
  return DenseElementsAttr::get(type, llvm::ArrayRef({0.0, 1.0, 1.0, 0.0}));
}

// RXOp

Value RXOp::getQubit(size_t i) {
  if (i != 0) {
    llvm_unreachable("RXOp has only one qubit");
  }
  return getQubitIn();
}

Value RXOp::getTarget(size_t i) {
  if (i != 0) {
    llvm_unreachable("RXOp has only one target qubit");
  }
  return getQubitIn();
}

Value RXOp::getPosControl(size_t /*i*/) {
  llvm_unreachable("RXOp does not have controls");
}

Value RXOp::getNegControl(size_t /*i*/) {
  llvm_unreachable("RXOp does not have controls");
}

ParameterDescriptor RXOp::getParameter(size_t i) {
  if (i != 0) {
    llvm_unreachable("RXOp has only one parameter");
  }
  return {getAngleStaticAttr(), getAngleDynamic()};
}

bool RXOp::hasStaticUnitary() { return getAngleStatic().has_value(); }

DenseElementsAttr RXOp::tryGetStaticMatrix() {
  if (!hasStaticUnitary()) {
    return nullptr;
  }
  auto* ctx = getContext();
  auto type = RankedTensorType::get({2, 2}, Float64Type::get(ctx));
  const auto angle = getAngleStatic().value().convertToDouble();
  const std::complex<double> c(cos(angle / 2), 0);
  const std::complex<double> s(0, -sin(angle / 2));
  return DenseElementsAttr::get(type, llvm::ArrayRef({c, s, s, c}));
}

LogicalResult RXOp::verify() {
  if (getAngleStatic().has_value() && getAngleDynamic())
    return emitOpError("cannot specify both static and dynamic angle");
  if (!getAngleStatic().has_value() && !getAngleDynamic())
    return emitOpError("must specify either static or dynamic angle");
  return success();
}
