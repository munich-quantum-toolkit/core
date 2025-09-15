/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Package.hpp"
#include "ir/Definitions.hpp"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"

#include <algorithm>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Operation.h>

namespace mqt::ir::opt::helpers {

std::optional<qc::fp> mlirValueToFp(mlir::Value value);

template <typename T, typename Func>
std::optional<qc::fp> performMlirFloatBinaryOp(mlir::Value value, Func&& func) {
  if (auto op = value.getDefiningOp<T>()) {
    auto lhs = mlirValueToFp(op.getLhs());
    auto rhs = mlirValueToFp(op.getRhs());
    if (lhs && rhs) {
      return std::invoke(std::forward<Func>(func), *lhs, *rhs);
    }
  }
  return std::nullopt;
}

template <typename T, typename Func>
std::optional<qc::fp> performMlirFloatUnaryOp(mlir::Value value, Func&& func) {
  if (auto op = value.getDefiningOp<T>()) {
    if (auto operand = mlirValueToFp(op.getOperand())) {
      return std::invoke(std::forward<Func>(func), *operand);
    }
  }
  return std::nullopt;
}

inline std::optional<qc::fp> mlirValueToFp(mlir::Value value) {
  if (auto op = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto attr = llvm::dyn_cast<mlir::FloatAttr>(op.getValue())) {
      return attr.getValueAsDouble();
    }
    return std::nullopt;
  }
  if (auto result = performMlirFloatUnaryOp<mlir::arith::NegFOp>(
          value, [](qc::fp a) { return -a; })) {
    return result;
  }
  if (auto result = performMlirFloatUnaryOp<mlir::arith::ExtFOp>(
          value, [](qc::fp a) { return a; })) {
    return result;
  }
  if (auto result = performMlirFloatUnaryOp<mlir::arith::TruncFOp>(
          value, [](qc::fp a) { return a; })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::MaxNumFOp>(
          value, [](qc::fp a, qc::fp b) { return std::max(a, b); })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::MaximumFOp>(
          value, [](qc::fp a, qc::fp b) { return std::max(a, b); })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::MinNumFOp>(
          value, [](qc::fp a, qc::fp b) { return std::min(a, b); })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::MinimumFOp>(
          value, [](qc::fp a, qc::fp b) { return std::min(a, b); })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::RemFOp>(
          value, [](qc::fp a, qc::fp b) { return std::fmod(a, b); })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::AddFOp>(
          value, [](qc::fp a, qc::fp b) { return a + b; })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::MulFOp>(
          value, [](qc::fp a, qc::fp b) { return a + b; })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::DivFOp>(
          value, [](qc::fp a, qc::fp b) { return a + b; })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::SubFOp>(
          value, [](qc::fp a, qc::fp b) { return a + b; })) {
    return result;
  }
  return std::nullopt;
}

[[nodiscard]] inline std::vector<qc::fp> getParameters(UnitaryInterface op) {
  std::vector<qc::fp> parameters;
  for (auto&& param : op.getParams()) {
    if (auto value = helpers::mlirValueToFp(param)) {
      parameters.push_back(*value);
    }
  }
  return parameters;
}

[[nodiscard]] inline qc::OpType getQcType(UnitaryInterface op) {
  try {
    const std::string type = op->getName().stripDialect().str();
    return qc::opTypeFromString(type);
  } catch (const std::invalid_argument& /*exception*/) {
    return qc::OpType::None;
  }
}

[[nodiscard]] inline bool isSingleQubitOperation(UnitaryInterface op) {
  auto&& inQubits = op.getInQubits();
  auto&& outQubits = op.getOutQubits();
  bool isSingleQubitOp =
      inQubits.size() == 1 && outQubits.size() == 1 && !op.isControlled();
  assert(isSingleQubitOp == qc::isSingleQubitGate(getQcType(op)));
  return isSingleQubitOp;
}

[[nodiscard]] inline bool isTwoQubitOperation(UnitaryInterface op) {
  auto&& inQubits = op.getInQubits();
  auto&& inPosCtrlQubits = op.getPosCtrlInQubits();
  auto&& inNegCtrlQubits = op.getNegCtrlInQubits();
  auto inQubitSize =
      inQubits.size() + inPosCtrlQubits.size() + inNegCtrlQubits.size();
  auto&& outQubits = op.getOutQubits();
  auto&& outPosCtrlQubits = op.getPosCtrlInQubits();
  auto&& outNegCtrlQubits = op.getNegCtrlInQubits();
  auto outQubitSize =
      outQubits.size() + outPosCtrlQubits.size() + outNegCtrlQubits.size();
  bool isTwoQubitOp = inQubitSize == 2 && outQubitSize == 2;
  assert(isTwoQubitOp == qc::isTwoQubitGate(getQcType(op)));
  return isTwoQubitOp;
}

[[nodiscard]] inline std::optional<dd::TwoQubitGateMatrix>
getUnitaryMatrix(UnitaryInterface op) {
  auto type = getQcType(op);
  auto parameters = getParameters(op);

  if (isTwoQubitOperation(op)) {
    return dd::opToTwoQubitGateMatrix(type, parameters);
  }
  return std::nullopt;
}

[[nodiscard]] inline dd::GateMatrix multiply(std::complex<qc::fp> factor,
                                             dd::GateMatrix matrix) {
  return {factor * matrix.at(0), factor * matrix.at(1), factor * matrix.at(2),
          factor * matrix.at(3)};
}

[[nodiscard]] inline dd::TwoQubitGateMatrix
kroneckerProduct(dd::GateMatrix lhs, dd::GateMatrix rhs) {
  return {multiply(lhs.at(0), rhs), multiply(lhs.at(1), rhs),
          multiply(lhs.at(2), rhs), multiply(lhs.at(3), rhs)};
}

[[nodiscard]] inline dd::TwoQubitGateMatrix
multiply(dd::TwoQubitGateMatrix lhs, dd::TwoQubitGateMatrix rhs) {
  return {};
}
} // namespace mqt::ir::opt::helpers
