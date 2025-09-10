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

[[nodiscard]] inline std::optional<dd::GateMatrix>
getUnitaryMatrix(UnitaryInterface op) {
  auto type = getQcType(op);
  auto parameters = getParameters(op);

  if (isSingleQubitOperation(op)) {
    return dd::opToSingleQubitGateMatrix(type, parameters);
  }
  return std::nullopt;
}

[[nodiscard]] inline dd::GateMatrix multiply(dd::GateMatrix lhs,
                                             dd::GateMatrix rhs) {
  return {
    lhs.at(0) * rhs.at(0) + lhs.at(1) * rhs.at(2),
    lhs.at(0) * rhs.at(1) + lhs.at(1) * rhs.at(3),
    lhs.at(2) * rhs.at(0) + lhs.at(3) * rhs.at(2),
    lhs.at(2) * rhs.at(1) + lhs.at(3) * rhs.at(3),
  };
}
} // namespace mqt::ir::opt::helpers
