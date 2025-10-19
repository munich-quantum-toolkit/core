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

namespace mqt::ir::opt {
  using fp = long double;
  using qfp = std::complex<fp>;
  using diagonal4x4 = std::array<qfp, 4>;
  using rdiagonal4x4 = std::array<fp, 4>;
  using vector2d = std::vector<qfp>;
  using matrix2x2 = std::array<qfp, 4>;
  using matrix4x4 = std::array<qfp, 16>;
  using rmatrix4x4 = std::array<fp, 16>;
} // namespace mqt::ir::opt

namespace mqt::ir::opt::helpers {

// TODO: remove
template <std::size_t N>
void print(std::array<std::complex<fp>, N> matrix) {
  int i{};
  for (auto&& a : matrix) {
    std::cerr << std::setprecision(50) << a.real() << 'i' << a.imag() << ' ';
    if (++i % 4 == 0) {
      llvm::errs() << '\n';
    }
  }
  llvm::errs() << '\n';
}

inline auto flatten(const dd::TwoQubitGateMatrix& matrix) {
  std::array<std::complex<fp>, 16> result;
  for (std::size_t i = 0; i < result.size(); ++i) {
    result[i] = matrix[i / 4][i % 4];
  }
  return result;
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

[[nodiscard]] inline std::vector<fp> getParameters(UnitaryInterface op) {
  std::vector<fp> parameters;
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

template <typename T, std::size_t N>
T kahanSum(const std::array<T, N>& values) {
  auto sum = T{};
  auto c = T{}; // Compensation for lost low-order bits

  for (auto&& value : values) {
    auto y = value - c; // Correct for error so far
    auto t = sum + y;   // Add the value to the running sum
    c = (t - sum) - y;  // Recompute the error
    sum = t;
  }

  return sum;
}

// Modify the matrix multiplication to use Kahan summation
template <typename T, std::size_t N>
std::array<T, N> matrixMultiplyWithKahan(const std::array<T, N>& lhs,
                                         const std::array<T, N>& rhs) {
  std::array<T, N> result;

  const std::size_t n = std::sqrt(N);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      std::array<T, n> terms;
      for (size_t k = 0; k < n; ++k) {
        terms[k] = lhs[i * n + k] * rhs[k * n + j];
      }
      result[i * n + j] = kahanSum(terms);
    }
  }
  return result;
}

[[nodiscard]] inline matrix2x2 multiply(qfp factor,
                                             const matrix2x2& matrix) {
  return {factor * matrix.at(0), factor * matrix.at(1), factor * matrix.at(2),
          factor * matrix.at(3)};
}

template <typename T, std::size_t N>
[[nodiscard]] inline auto multiply(const std::array<T, N>& lhs,
                                   const std::array<T, N>& rhs) {
  // return matrixMultiplyWithKahan(lhs, rhs);
  std::array<T, N> result{{T{}}};
  const int n = std::sqrt(N);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        result[i * n + j] += lhs[i * n + k] * rhs[k * n + j];
      }
    }
  }
  return result;
}

template <typename T>
[[nodiscard]] inline auto LUdecomposition(std::array<T, 16> matrix) {
  std::array<T, 16> L{};
  std::array<T, 16> U{};
  int rowPermutations = 0;

  for (int i = 0; i < 4; i++) {
    // --- Partial pivoting: find max row in column i ---
    int pivotRow = i;
    auto maxVal = matrix[i * 4 + i];

    for (int r = i + 1; r < 4; r++) {
      auto val = matrix[r * 4 + i];
      if (std::abs(val) > std::abs(maxVal)) {
        maxVal = val;
        pivotRow = r;
      }
    }

    // --- Swap rows in matrix if needed ---
    if (pivotRow != i) {
      for (int col = 0; col < 4; ++col) {
        std::swap(matrix[i * 4 + col], matrix[pivotRow * 4 + col]);
      }
      ++rowPermutations;
    }

    // --- Compute L matrix (column-wise) ---
    for (int j = 0; j < 4; j++) {
      if (j < i)
        L[j * 4 + i] = 0;
      else {
        L[j * 4 + i] = matrix[j * 4 + i];
        for (int k = 0; k < i; k++) {
          L[j * 4 + i] -= L[j * 4 + k] * U[k * 4 + i];
        }
      }
    }

    // --- Compute U matrix (row-wise) ---
    for (int j = 0; j < 4; j++) {
      if (j < i)
        U[i * 4 + j] = 0;
      else if (j == i)
        U[i * 4 + j] = 1; // Diagonal of U is set to 1
      else {
        U[i * 4 + j] = matrix[i * 4 + j] / L[i * 4 + i];
        for (int k = 0; k < i; k++) {
          U[i * 4 + j] -= (L[i * 4 + k] * U[k * 4 + j]) / L[i * 4 + i];
        }
      }
    }
  }

  return std::make_tuple(L, U, rowPermutations);
}
} // namespace mqt::ir::opt::helpers
