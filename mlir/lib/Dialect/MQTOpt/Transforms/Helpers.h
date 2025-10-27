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
using fp = double;
using qfp = std::complex<fp>;
using diagonal4x4 = std::array<qfp, 4>;
using rdiagonal4x4 = std::array<fp, 4>;
using vector2d = std::vector<qfp>;
using matrix2x2 = std::array<qfp, 4>;
using matrix4x4 = std::array<qfp, 16>;
using rmatrix4x4 = std::array<fp, 16>;

constexpr qfp C_ZERO{0., 0.};
constexpr qfp C_ONE{1., 0.};
constexpr qfp C_M_ONE{-1., 0.};
constexpr qfp IM{0., 1.};
constexpr qfp M_IM{0., -1.};

} // namespace mqt::ir::opt

namespace mqt::ir::opt::helpers {

// TODO: remove
template <std::size_t N> void print(std::array<std::complex<fp>, N> matrix, std::string s = "") {
  int i{};
  if (!s.empty()) {
    llvm::errs() << "=== " << s << " ===\n";
  }
  for (auto&& a : matrix) {
    std::cerr << std::setprecision(17) << a.real() << 'i' << a.imag() << ' ';
    if (++i % 4 == 0) {
      llvm::errs() << '\n';
    }
  }
  llvm::errs() << '\n';
}

template <std::size_t N> void print(std::array<fp, N> matrix, std::string s = "") {
  int i{};
  if (!s.empty()) {
    llvm::errs() << "=== " << s << " ===\n";
  }
  for (auto&& a : matrix) {
    std::cerr << std::setprecision(17) << a << ' ';
    if (++i % 4 == 0) {
      llvm::errs() << '\n';
    }
  }
  llvm::errs() << '\n';
}

template <std::size_t N> void print(std::array<std::size_t, N> matrix, std::string s = "") {
  int i{};
  if (!s.empty()) {
    llvm::errs() << "=== " << s << " ===\n";
  }
  for (auto&& a : matrix) {
    std::cerr << a << ' ';
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

template <typename Container>
[[nodiscard]] inline Container multiply(fp factor, Container matrix) {
  llvm::transform(matrix, std::begin(matrix),
                  [&](auto&& x) { return factor * x; });
  return matrix;
}

template <typename Container>
[[nodiscard]] inline Container multiply(qfp factor, Container matrix) {
  llvm::transform(matrix, std::begin(matrix),
                  [&](auto&& x) { return factor * x; });
  return matrix;
}

template <typename T, std::size_t N>
[[nodiscard]] inline auto multiply(const std::array<T, N>& lhs,
                                   const std::array<T, N>& rhs) {
  std::array<T, N> result{};
  const int n = std::sqrt(lhs.size());
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
[[nodiscard]] inline auto multiply(const std::vector<T>& lhs,
                                   const std::vector<T>& rhs, int columnsLhs) {
  int rowsLhs = lhs.size() / columnsLhs;
  int rowsRhs = columnsLhs;
  int columnsRhs = rhs.size() / rowsRhs;
  assert(rowsLhs * columnsLhs == lhs.size());
  assert(rowsRhs * columnsRhs == rhs.size());

  std::vector<T> result(rowsLhs * columnsRhs, T{});
  for (int i = 0; i < rowsLhs; i++) {
    for (int j = 0; j < columnsRhs; j++) {
      for (int k = 0; k < columnsLhs; k++) {
        result[i * columnsRhs + j] +=
            lhs[i * columnsLhs + k] * rhs[k * columnsRhs + j];
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

template <typename Container>
using ValueType = typename std::remove_cvref_t<Container>::value_type;

template <int Offset = 0, typename Container>
static auto diagonal(Container&& matrix) {
  const int n = std::sqrt(matrix.size());
  auto result = [&]() {
    using T = std::remove_cvref_t<Container>;
    if constexpr (std::is_same_v<T, matrix4x4> && Offset == 0) {
      return diagonal4x4{};
    } else if constexpr (std::is_same_v<T, rmatrix4x4> && Offset == 0) {
      return rdiagonal4x4{};
    } else {
      return std::vector<ValueType<Container>>(n - std::abs(Offset));
    }
  }();
  for (std::size_t i = 0; i < result.size(); ++i) {
    auto x = Offset > 0 ? i + Offset : i;
    auto y = Offset < 0 ? i - Offset : i;
    result[i] = matrix[y * n + x];
  }
  return result;
}

template <typename Container>
auto submatrix(Container&& matrix, int rowStart, int columnStart, int numRows,
               int numColumns) {
  const int n = std::sqrt(matrix.size());
  assert((rowStart + numRows) <= n);
  assert((columnStart + numColumns) <= n);

  std::vector<ValueType<Container>> result(numRows * numColumns);
  for (int i = 0; i < numColumns; ++i) {
    for (int j = 0; j < numRows; ++j) {
      result[j * numColumns + i] = matrix[(rowStart + j) * n + (columnStart + i)];
    }
  }
  return result;
}

template <typename Lhs, typename Rhs>
auto assignSubmatrix(Lhs&& lhs, Rhs&& rhs, int rowStart, int columnStart,
                     int numRows, int numColumns) {
  const int n = std::sqrt(lhs.size());
  assert((rowStart + numRows) <= n);
  assert((columnStart + numColumns) <= n);
  assert(numColumns * numRows == rhs.size());

  for (int i = 0; i < numColumns; ++i) {
    for (int j = 0; j < numRows; ++j) {
      lhs[(rowStart + j) * n + (columnStart + i)] = rhs[j * numColumns + i];
    }
  }
}

template <typename... Args> inline auto dot(Args&&... args) {
  return helpers::multiply(std::forward<Args>(args)...);
}

template <typename Lhs, typename Rhs>
inline auto vectorsDot(Lhs&& lhs, Rhs&& rhs) {
  return std::inner_product(lhs.begin(), lhs.end(), rhs.begin(),
                            typename Lhs::value_type{});
}

template <typename Lhs, typename Rhs> auto add(Lhs lhs, Rhs&& rhs) {
  assert(lhs.size() == rhs.size());
  for (int i = 0; i < lhs.size(); ++i) {
    lhs[i] += rhs[i];
  }
  return lhs;
}

inline matrix2x2 transpose(const matrix2x2& matrix) {
  return {matrix[0 * 2 + 0], matrix[1 * 2 + 0], matrix[0 * 2 + 1],
          matrix[1 * 2 + 1]};
}

template <typename Container> auto transpose(Container&& matrix) {
  const std::size_t n = std::sqrt(matrix.size());
  auto result{matrix};
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      result[j * n + i] = matrix[i * n + j];
    }
  }
  return result;
}

template <typename T> auto conj(T&& x) {
  using U = std::remove_cvref_t<T>;
  if constexpr (std::is_same_v<U, qfp>) {
    return std::conj(x);
  } else if constexpr (std::is_same_v<U, fp>) {
    return x;
  } else {
    static_assert(!sizeof(U), "Unimplemented case for helpers::conj");
  }
}

template <typename Container> auto conjugate(Container matrix) {
  llvm::transform(matrix, matrix.begin(), [](auto&& x) { return conj(x); });
  return matrix;
}

template <typename Container>
inline auto transpose_conjugate(Container&& matrix) {
  auto result = transpose(matrix);
  return conjugate(result);
}

template<typename T>
inline T determinant(const std::array<T, 4>& mat) {
  return mat[0] * mat[3] - mat[1] * mat[2];
}

template<typename T>
inline T determinant(const std::array<T, 9>& mat) {
  return mat[0] * (mat[4] * mat[8] - mat[5] * mat[7]) -
         mat[1] * (mat[3] * mat[8] - mat[5] * mat[6]) +
         mat[2] * (mat[3] * mat[7] - mat[4] * mat[6]);
}

inline std::array<qfp, 9> get3x3Submatrix(const matrix4x4& mat,
                                          int rowToBeRemoved,
                                          int columnToBeRemoved) {
  std::array<std::complex<fp>, 9> result;
  int subIndex = 0;
  for (int i = 0; i < 4; ++i) {
    if (i != rowToBeRemoved) {
      for (int j = 0; j < 4; ++j) {
        if (j != columnToBeRemoved) {
          result[subIndex++] = mat[i * 4 + j];
        }
      }
    }
  }
  return result;
}

template<typename T>
inline T determinant(const std::array<T, 16>& mat) {
  auto [l, u, rowPermutations] = helpers::LUdecomposition(mat);
  T det = 1.0;
  for (int i = 0; i < 4; ++i) {
    det *= l[i * 4 + i];
  }

  if (rowPermutations % 2 != 0) {
    det = -det;
  }
  return det;

  // auto det = -C_ZERO;
  // for (int column = 0; column < 4; ++column) {
  //   auto submatrix = get3x3Submatrix(mat, 0, column);
  //   auto subDet = determinant(submatrix);
  //   auto tmp = mat[0 * 4 + column] * subDet;
  //   if (column % 2 == 0 &&
  //       tmp !=
  //           C_ZERO) { // TODO: better way to get negative 0.0 in
  //           determinant?
  //     det += tmp;
  //   } else if (tmp != -C_ZERO) {
  //     det -= tmp;
  //   }
  // }
  // return det;
}

template <typename T>
inline std::array<T, 16> from(const std::array<T, 4>& first_quadrant,
                              const std::array<T, 4>& second_quadrant,
                              const std::array<T, 4>& third_quadrant,
                              const std::array<T, 4>& fourth_quadrant) {
  return {
      first_quadrant[0 * 2 + 0],  first_quadrant[0 * 2 + 1],
      second_quadrant[0 * 2 + 0], second_quadrant[0 * 2 + 1],
      first_quadrant[1 * 2 + 0],  first_quadrant[1 * 2 + 1],
      second_quadrant[1 * 2 + 0], second_quadrant[1 * 2 + 1],
      third_quadrant[0 * 2 + 0],  third_quadrant[0 * 2 + 1],
      fourth_quadrant[0 * 2 + 0], fourth_quadrant[0 * 2 + 1],
      third_quadrant[1 * 2 + 0],  third_quadrant[1 * 2 + 1],
      fourth_quadrant[1 * 2 + 0], fourth_quadrant[1 * 2 + 1],
  };
}

template <typename T> inline auto toArray4(const std::vector<T>& vec) {
  std::array<T, 4> result;
  assert(vec.size() == result.size());
  for (std::size_t i = 0; i < vec.size(); ++i) {
    result[i] = vec[i];
  }
  return result;
}

template <typename T>
inline std::array<T, 16> kroneckerProduct(const std::array<T, 4>& lhs,
                                          const std::array<T, 4>& rhs) {
  return from(multiply(lhs[0 * 2 + 0], rhs), multiply(lhs[0 * 2 + 1], rhs),
              multiply(lhs[1 * 2 + 0], rhs), multiply(lhs[1 * 2 + 1], rhs));
}

} // namespace mqt::ir::opt::helpers
