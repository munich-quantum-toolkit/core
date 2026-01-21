/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "ir/Definitions.hpp"
#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <Eigen/Core>        // NOLINT(misc-include-cleaner)
#include <Eigen/Eigenvalues> // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <cmath>
#include <complex>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <unsupported/Eigen/KroneckerProduct> // TODO: unstable, NOLINT(misc-include-cleaner)

namespace mlir::qco {
using fp = qc::fp;
using qfp = std::complex<fp>;
// NOLINTBEGIN(misc-include-cleaner)
using matrix2x2 = Eigen::Matrix2<qfp>;
using matrix4x4 = Eigen::Matrix4<qfp>;
using rmatrix4x4 = Eigen::Matrix4<fp>;
using diagonal4x4 = Eigen::Vector<qfp, 4>;
using rdiagonal4x4 = Eigen::Vector<fp, 4>;
// NOLINTEND(misc-include-cleaner)

constexpr qfp C_ZERO{0., 0.};
constexpr qfp C_ONE{1., 0.};
constexpr qfp C_M_ONE{-1., 0.};
constexpr qfp IM{0., 1.};
constexpr qfp M_IM{0., -1.};

} // namespace mlir::qco

namespace mlir::qco::helpers {

std::optional<fp> mlirValueToFp(mlir::Value value);

template <typename T, typename Func>
[[nodiscard]] std::optional<fp> performMlirFloatBinaryOp(mlir::Value value,
                                                         Func&& func) {
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
[[nodiscard]] std::optional<fp> performMlirFloatUnaryOp(mlir::Value value,
                                                        Func&& func) {
  if (auto op = value.getDefiningOp<T>()) {
    if (auto operand = mlirValueToFp(op.getOperand())) {
      return std::invoke(std::forward<Func>(func), *operand);
    }
  }
  return std::nullopt;
}

[[nodiscard]] inline std::optional<fp> mlirValueToFp(mlir::Value value) {
  if (auto op = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto attr = llvm::dyn_cast<mlir::FloatAttr>(op.getValue())) {
      return attr.getValueAsDouble();
    }
    return std::nullopt;
  }
  if (auto result = performMlirFloatUnaryOp<mlir::arith::NegFOp>(
          value, [](fp a) { return -a; })) {
    return result;
  }
  if (auto result = performMlirFloatUnaryOp<mlir::arith::ExtFOp>(
          value, [](fp a) { return a; })) {
    return result;
  }
  if (auto result = performMlirFloatUnaryOp<mlir::arith::TruncFOp>(
          value, [](fp a) { return a; })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::MaxNumFOp>(
          value, [](fp a, fp b) { return std::max(a, b); })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::MaximumFOp>(
          value, [](fp a, fp b) { return std::max(a, b); })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::MinNumFOp>(
          value, [](fp a, fp b) { return std::min(a, b); })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::MinimumFOp>(
          value, [](fp a, fp b) { return std::min(a, b); })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::RemFOp>(
          value, [](fp a, fp b) { return std::fmod(a, b); })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::AddFOp>(
          value, [](fp a, fp b) { return a + b; })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::MulFOp>(
          value, [](fp a, fp b) { return a * b; })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::DivFOp>(
          value, [](fp a, fp b) { return a / b; })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::SubFOp>(
          value, [](fp a, fp b) { return a - b; })) {
    return result;
  }
  return std::nullopt;
}

[[nodiscard]] inline llvm::SmallVector<fp, 3>
getParameters(UnitaryOpInterface op) {
  llvm::SmallVector<fp, 3> parameters;
  for (std::size_t i = 0; i < op.getNumParams(); ++i) {
    if (auto value = helpers::mlirValueToFp(op.getParameter(i))) {
      parameters.push_back(*value);
    }
  }
  return parameters;
}

[[nodiscard]] inline qc::OpType getQcType(UnitaryOpInterface op) {
  try {
    const std::string type = op->getName().stripDialect().str();
    return qc::opTypeFromString(type);
  } catch (const std::invalid_argument& /*exception*/) {
    return qc::OpType::None;
  }
}

[[nodiscard]] inline bool isSingleQubitOperation(UnitaryOpInterface op) {
  return op.isSingleQubit();
}

[[nodiscard]] inline bool isTwoQubitOperation(UnitaryOpInterface op) {
  return op.isTwoQubit();
}

// NOLINTBEGIN(misc-include-cleaner)
template <typename T>
[[nodiscard]] inline Eigen::Matrix4<T>
kroneckerProduct(const Eigen::Matrix2<T>& lhs, const Eigen::Matrix2<T>& rhs) {
  return Eigen::kroneckerProduct(lhs, rhs);
}

template <typename T, int N, int M>
[[nodiscard]] inline auto selfAdjointEvd(Eigen::Matrix<T, N, M> a) {
  Eigen::SelfAdjointEigenSolver<decltype(a)> s;
  s.compute(a); // TODO: computeDirect is faster
  auto vecs = s.eigenvectors().eval();
  auto vals = s.eigenvalues();
  return std::make_pair(vecs, vals);
}

template <typename T, int N, int M>
[[nodiscard]] bool isUnitaryMatrix(const Eigen::Matrix<T, N, M>& matrix) {
  return (matrix.transpose().conjugate() * matrix).isIdentity();
}
// NOLINTEND(misc-include-cleaner)

[[nodiscard]] inline fp remEuclid(fp a, fp b) {
  auto r = std::fmod(a, b);
  return (r < 0.0) ? r + std::abs(b) : r;
}

// Wrap angle into interval [-π,π). If within atol of the endpoint, clamp
// to -π
[[nodiscard]] inline fp mod2pi(fp angle, fp angleZeroEpsilon = 1e-13) {
  // remEuclid() isn't exactly the same as Python's % operator, but
  // because the RHS here is a constant and positive it is effectively
  // equivalent for this case
  auto wrapped = remEuclid(angle + qc::PI, qc::TAU) - qc::PI;
  if (std::abs(wrapped - qc::PI) < angleZeroEpsilon) {
    return -qc::PI;
  }
  return wrapped;
}

[[nodiscard]] inline fp traceToFidelity(const qfp& x) {
  auto xAbs = std::abs(x);
  return (4.0 + xAbs * xAbs) / 20.0;
}

[[nodiscard]] inline std::size_t getComplexity(qc::OpType type,
                                               std::size_t numOfQubits) {
  if (numOfQubits > 1) {
    constexpr std::size_t multiQubitFactor = 10;
    return (numOfQubits - 1) * multiQubitFactor;
  }
  if (type == qc::GPhase) {
    return 2;
  }
  return 1;
}

} // namespace mlir::qco::helpers
