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

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "ir/operations/OpType.hpp"

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/unsupported/Eigen/KroneckerProduct> // TODO: unstable
#include <iomanip> // TODO: remove
#include <iostream> // TODO: remove
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Operation.h>

namespace mqt::ir::opt {
using fp = double;
using qfp = std::complex<fp>;
using matrix2x2 = Eigen::Matrix2<qfp>;
using matrix4x4 = Eigen::Matrix4<qfp>;
using rmatrix4x4 = Eigen::Matrix4<fp>;
using diagonal4x4 = Eigen::Vector<qfp, 4>;
using rdiagonal4x4 = Eigen::Vector<fp, 4>;
;

constexpr qfp C_ZERO{0., 0.};
constexpr qfp C_ONE{1., 0.};
constexpr qfp C_M_ONE{-1., 0.};
constexpr qfp IM{0., 1.};
constexpr qfp M_IM{0., -1.};

} // namespace mqt::ir::opt

namespace mqt::ir::opt::helpers {

inline void print(std::size_t x) { std::cerr << x; }
inline void print(fp x) { std::cerr << x; }

inline void print(qfp x) {
  std::cerr << std::setprecision(17) << x.real() << 'i' << x.imag();
}

// TODO: remove
template <typename T, int N, int M>
void print(Eigen::Matrix<T, N, M> matrix, const std::string& s = "",
           bool force = false) {
  if (!force) {
    return;
}
  if (!s.empty()) {
    llvm::errs() << "=== " << s << " ===\n";
  }
  std::cerr << matrix;
  llvm::errs() << '\n';
}

template <typename T>
void print(T matrix, const std::string& s = "", bool force = false) {
  if (!force) {
    return;
}
  if (!s.empty()) {
    llvm::errs() << "=== " << s << " ===\n";
  }

  for (auto&& a : matrix) {
    print(a);
    std::cerr << ' ';
  }
  llvm::errs() << '\n';
}

std::optional<fp> mlirValueToFp(mlir::Value value);

template <typename T, typename Func>
std::optional<fp> performMlirFloatBinaryOp(mlir::Value value, Func&& func) {
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
std::optional<fp> performMlirFloatUnaryOp(mlir::Value value, Func&& func) {
  if (auto op = value.getDefiningOp<T>()) {
    if (auto operand = mlirValueToFp(op.getOperand())) {
      return std::invoke(std::forward<Func>(func), *operand);
    }
  }
  return std::nullopt;
}

inline std::optional<fp> mlirValueToFp(mlir::Value value) {
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
          value, [](fp a, fp b) { return a + b; })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::DivFOp>(
          value, [](fp a, fp b) { return a + b; })) {
    return result;
  }
  if (auto result = performMlirFloatBinaryOp<mlir::arith::SubFOp>(
          value, [](fp a, fp b) { return a + b; })) {
    return result;
  }
  return std::nullopt;
}

[[nodiscard]] inline llvm::SmallVector<fp, 3>
getParameters(UnitaryInterface op) {
  llvm::SmallVector<fp, 3> parameters;
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
  return isTwoQubitOp;
}

template <typename T>
inline Eigen::Matrix4<T> kroneckerProduct(const Eigen::Matrix2<T>& lhs,
                                          const Eigen::Matrix2<T>& rhs) {
  Eigen::Matrix4<T> result;
  Eigen::KroneckerProduct kroneckerProduct{lhs, rhs};
  kroneckerProduct.evalTo(result);
  return result;
}

template<typename T, int N, int M>
inline auto selfAdjointEvd(Eigen::Matrix<T, N, M> a) {
  Eigen::SelfAdjointEigenSolver<decltype(a)> s;
  std::cerr << "=EigIN==\n" << a << "\n========\n" << '\n';
  s.compute(a); // TODO: computeDirect is faster
  auto vecs = s.eigenvectors().eval();
  auto vals = s.eigenvalues();
  std::cerr << "=Eigen==\n" << vecs << "\n========\n" << '\n';
  std::cerr << "=Eigen==\n" << vals << "\n========\n" << '\n';
  return std::make_pair(vecs, vals);
}

} // namespace mqt::ir::opt::helpers
