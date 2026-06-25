/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>

namespace mlir::qco {

/// Returns true if @p lhs and @p rhs differ by at most @p tol (complex
/// modulus).
[[nodiscard]] static bool entryIsApprox(const Complex& lhs, const Complex& rhs,
                                        const double tol) {
  return std::abs(lhs - rhs) <= tol;
}

/// Returns true if every entry pair differs by at most @p tol (complex
/// modulus).
[[nodiscard]] static bool entriesAreApprox(ArrayRef<Complex> lhs,
                                           ArrayRef<Complex> rhs, double tol) {
  return llvm::equal(lhs, rhs, [tol](const Complex& a, const Complex& b) {
    return entryIsApprox(a, b, tol);
  });
}

/// Writes the conjugate transpose of @p in into @p out (square, row-major).
static void adjointInto(ArrayRef<Complex> in, MutableArrayRef<Complex> out,
                        const std::size_t dim) {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      out[(row * dim) + col] = std::conj(in[(col * dim) + row]);
    }
  }
}

template <std::size_t Dim, std::size_t Size>
static void assignFixedImpl(std::int64_t& dim, SmallVector<Complex>& data,
                            const std::array<Complex, Size>& src) {
  dim = static_cast<std::int64_t>(Dim);
  data.assign(src.begin(), src.end());
}

template <std::size_t Dim, std::size_t Size>
[[nodiscard]] static bool
isApproxFixedImpl(const std::int64_t dim, const SmallVector<Complex>& data,
                  const std::array<Complex, Size>& other, const double tol) {
  if (std::cmp_not_equal(dim, Dim)) {
    return false;
  }
  return entriesAreApprox(data, other, tol);
}

template <std::size_t Dim, std::size_t Size>
[[nodiscard]] static bool
assignFromDynamicImpl(const DynamicMatrix& src,
                      std::array<Complex, Size>& dst) {
  if (src.rows() != static_cast<std::int64_t>(Dim) ||
      src.cols() != static_cast<std::int64_t>(Dim)) {
    return false;
  }
  for (std::size_t row = 0; row < Dim; ++row) {
    for (std::size_t col = 0; col < Dim; ++col) {
      dst[(row * Dim) + col] =
          src(static_cast<std::int64_t>(row), static_cast<std::int64_t>(col));
    }
  }
  return true;
}

/// Writes the row-major product `lhs * rhs` into @p out (2x2, fully unrolled).
static void multiply2x2(const ArrayRef<Complex> lhs,
                        const ArrayRef<Complex> rhs,
                        const MutableArrayRef<Complex> out) {
  assert(lhs.size() == Matrix2x2::K_SIZE_AT_COMPILE_TIME &&
         rhs.size() == Matrix2x2::K_SIZE_AT_COMPILE_TIME &&
         out.size() == Matrix2x2::K_SIZE_AT_COMPILE_TIME);
  out[0] = lhs[0] * rhs[0] + lhs[1] * rhs[2];
  out[1] = lhs[0] * rhs[1] + lhs[1] * rhs[3];
  out[2] = lhs[2] * rhs[0] + lhs[3] * rhs[2];
  out[3] = lhs[2] * rhs[1] + lhs[3] * rhs[3];
}

/// Writes the row-major product `lhs * rhs` into @p out (4x4, unrolled rows).
static void multiply4x4(const ArrayRef<Complex> lhs,
                        const ArrayRef<Complex> rhs,
                        const MutableArrayRef<Complex> out) {
  assert(lhs.size() == Matrix4x4::K_SIZE_AT_COMPILE_TIME &&
         rhs.size() == Matrix4x4::K_SIZE_AT_COMPILE_TIME &&
         out.size() == Matrix4x4::K_SIZE_AT_COMPILE_TIME);
  for (std::size_t row = 0; row < Matrix4x4::K_ROWS; ++row) {
    const std::size_t rowBase = row * Matrix4x4::K_COLS;
    const Complex& a0 = lhs[rowBase + 0];
    const Complex& a1 = lhs[rowBase + 1];
    const Complex& a2 = lhs[rowBase + 2];
    const Complex& a3 = lhs[rowBase + 3];
    out[rowBase + 0] = a0 * rhs[0] + a1 * rhs[4] + a2 * rhs[8] + a3 * rhs[12];
    out[rowBase + 1] = a0 * rhs[1] + a1 * rhs[5] + a2 * rhs[9] + a3 * rhs[13];
    out[rowBase + 2] = a0 * rhs[2] + a1 * rhs[6] + a2 * rhs[10] + a3 * rhs[14];
    out[rowBase + 3] = a0 * rhs[3] + a1 * rhs[7] + a2 * rhs[11] + a3 * rhs[15];
  }
}

/// Returns true if @p data is approximately the @p dim x @p dim identity
/// matrix.
[[nodiscard]] static bool isIdentityEntries(ArrayRef<Complex> data,
                                            const std::size_t dim,
                                            const double tol) {
  assert(data.size() >= dim * dim);
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      const Complex& entry = data[(row * dim) + col];
      if (row == col) {
        if (!entryIsApprox(entry, Complex{1.0, 0.0}, tol)) {
          return false;
        }
      } else if (std::abs(entry) > tol) {
        return false;
      }
    }
  }
  return true;
}

/// Returns @p dim as `size_t` after asserting it is non-negative and squarable.
[[nodiscard]] static std::size_t checkedDim(const std::int64_t dim) {
  assert(dim >= 0 && "DynamicMatrix dimension must be non-negative");
  const auto udim = static_cast<std::size_t>(dim);
  assert(udim == 0 || udim <= std::numeric_limits<std::size_t>::max() / udim);
  return udim;
}

[[nodiscard]] static std::size_t checkedStorageSize(const std::int64_t dim) {
  const auto udim = checkedDim(dim);
  return udim * udim;
}

/// Returns `2^numQubits` as `int64_t` after checking it fits.
[[nodiscard]] static std::int64_t
checkedHilbertDim(const std::size_t numQubits) {
  assert(numQubits < std::numeric_limits<std::int64_t>::digits &&
         "Hilbert-space dimension must fit in int64_t");
  return std::int64_t{static_cast<int64_t>(std::uint64_t{1} << numQubits)};
}

static void validateCornerDims(const std::int64_t matrixDim,
                               const std::int64_t blockDim) {
  assert(matrixDim >= 0 && blockDim >= 0 && blockDim <= matrixDim &&
         "block must fit in the bottom-right corner of the matrix");
  std::ignore = checkedDim(matrixDim);
}

/// Copies @p blockData into the bottom-right @p blockDim x @p blockDim corner.
static void copyBottomRightCorner(const std::int64_t matrixDim,
                                  MutableArrayRef<Complex> matrixData,
                                  const std::int64_t blockDim,
                                  ArrayRef<Complex> blockData) {
  validateCornerDims(matrixDim, blockDim);
  assert(matrixData.size() >= checkedStorageSize(matrixDim));
  assert(blockData.size() >= checkedStorageSize(blockDim));
  const std::int64_t offset = matrixDim - blockDim;
  for (std::int64_t row = 0; row < blockDim; ++row) {
    for (std::int64_t col = 0; col < blockDim; ++col) {
      matrixData[static_cast<std::size_t>(((offset + row) * matrixDim) +
                                          offset + col)] =
          blockData[static_cast<std::size_t>((row * blockDim) + col)];
    }
  }
}

/**
 * @brief Returns the @p qubitIndex bit of a computational-basis label.
 *
 * Qubit 0 is the MSB of @p stateIndex, matching @ref Matrix4x4::kron and
 * @ref Matrix2x2::embedInNqubit.
 */
[[nodiscard]] static std::size_t qubitBitAt(const std::size_t stateIndex,
                                            const std::size_t numQubits,
                                            const std::size_t qubitIndex) {
  return (stateIndex >> (numQubits - 1 - qubitIndex)) & 1U;
}

/**
 * @brief True when row and col agree on every wire except @p skipA and @p
 * skipB.
 *
 * Used when embedding a gate: untouched qubits must match or the matrix entry
 * is zero. For a single-qubit embed, pass @p skipB = @p numQubits so only @p
 * skipA is skipped.
 */
[[nodiscard]] static bool otherQubitBitsMatch(const std::size_t row,
                                              const std::size_t col,
                                              const std::size_t numQubits,
                                              const std::size_t skipA,
                                              const std::size_t skipB) {
  for (std::size_t q = 0; q < numQubits; ++q) {
    if (q == skipA || q == skipB) {
      continue;
    }
    if (qubitBitAt(row, numQubits, q) != qubitBitAt(col, numQubits, q)) {
      return false;
    }
  }
  return true;
}

Matrix1x1 Matrix1x1::fromElements(const Complex m00) { return {m00}; }

Complex& Matrix1x1::operator()(const std::size_t row, const std::size_t col) {
  assert(row == 0 && col == 0);
  return value;
}

Complex Matrix1x1::operator()(const std::size_t row,
                              const std::size_t col) const {
  assert(row == 0 && col == 0);
  return value;
}

bool Matrix1x1::isApprox(const Matrix1x1& other, const double tol) const {
  return entryIsApprox(value, other.value, tol);
}

bool Matrix1x1::assignFrom(const DynamicMatrix& src) {
  if (src.rows() != 1 || src.cols() != 1) {
    return false;
  }
  value = src(0, 0);
  return true;
}

Matrix1x1 Matrix1x1::operator*(const Complex& scalar) const {
  return fromElements(value * scalar);
}

Matrix1x1& Matrix1x1::operator*=(const Complex& scalar) {
  value *= scalar;
  return *this;
}

Matrix1x1 Matrix1x1::adjoint() const { return fromElements(std::conj(value)); }

Matrix2x2 Matrix2x2::fromElements(const Complex& m00, const Complex& m01,
                                  const Complex& m10, const Complex& m11) {
  return {{m00, m01, m10, m11}};
}

Complex& Matrix2x2::operator()(const std::size_t row, const std::size_t col) {
  return data[(row * K_COLS) + col];
}

Complex Matrix2x2::operator()(const std::size_t row,
                              const std::size_t col) const {
  return data[(row * K_COLS) + col];
}

Matrix2x2 Matrix2x2::operator*(const Matrix2x2& rhs) const {
  Matrix2x2 out{};
  multiply2x2(data, rhs.data, out.data);
  return out;
}

void Matrix2x2::premultiplyBy(const Matrix2x2& lhs) { *this = lhs * *this; }

Matrix2x2 Matrix2x2::operator*(const Complex& scalar) const {
  Matrix2x2 out = *this;
  out *= scalar;
  return out;
}

Matrix2x2& Matrix2x2::operator*=(const Complex& scalar) {
  for (Complex& entry : data) {
    entry *= scalar;
  }
  return *this;
}

Matrix2x2 Matrix2x2::adjoint() const {
  return fromElements(std::conj(data[0]), std::conj(data[2]),
                      std::conj(data[1]), std::conj(data[3]));
}

Matrix2x2 Matrix2x2::transpose() const {
  return fromElements(data[0], data[2], data[1], data[3]);
}

Complex Matrix2x2::trace() const { return data[0] + data[3]; }

Complex Matrix2x2::determinant() const {
  return data[0] * data[3] - data[1] * data[2];
}

bool Matrix2x2::isIdentity(const double tol) const {
  return isIdentityEntries(data, K_ROWS, tol);
}

bool Matrix2x2::isApprox(const Matrix2x2& other, const double tol) const {
  return entriesAreApprox(data, other.data, tol);
}

bool Matrix2x2::assignFrom(const DynamicMatrix& src) {
  return assignFromDynamicImpl<K_ROWS, K_SIZE_AT_COMPILE_TIME>(src, data);
}

DynamicMatrix Matrix2x2::embedInNqubit(const std::size_t numQubits,
                                       const std::size_t qubitIndex) const {
  assert(qubitIndex < numQubits &&
         "Invalid qubit index for single-qubit embed");
  if (numQubits == 2) {
    return DynamicMatrix(embedInTwoQubit(qubitIndex));
  }
  const auto dim = checkedHilbertDim(numQubits);
  DynamicMatrix out(dim);
  const auto udim = static_cast<std::size_t>(dim);
  for (std::size_t row = 0; row < udim; ++row) {
    for (std::size_t col = 0; col < udim; ++col) {
      if (!otherQubitBitsMatch(row, col, numQubits, qubitIndex, numQubits)) {
        continue;
      }
      const std::size_t rowBit = qubitBitAt(row, numQubits, qubitIndex);
      const std::size_t colBit = qubitBitAt(col, numQubits, qubitIndex);
      out(static_cast<std::int64_t>(row), static_cast<std::int64_t>(col)) =
          (*this)(rowBit, colBit);
    }
  }
  return out;
}

Matrix4x4 Matrix2x2::embedInTwoQubit(const std::size_t qubitIndex) const {
  if (qubitIndex == 0) {
    return Matrix4x4::kron(*this, Matrix2x2::identity());
  }
  if (qubitIndex == 1) {
    return Matrix4x4::kron(Matrix2x2::identity(), *this);
  }
  llvm::reportFatalInternalError("Invalid qubit index for single-qubit embed");
}

Matrix4x4 Matrix4x4::fromElements(const Complex& m00, const Complex& m01,
                                  const Complex& m02, const Complex& m03,
                                  const Complex& m10, const Complex& m11,
                                  const Complex& m12, const Complex& m13,
                                  const Complex& m20, const Complex& m21,
                                  const Complex& m22, const Complex& m23,
                                  const Complex& m30, const Complex& m31,
                                  const Complex& m32, const Complex& m33) {
  return {{m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31,
           m32, m33}};
}

Complex& Matrix4x4::operator()(const std::size_t row, const std::size_t col) {
  return data[(row * K_COLS) + col];
}

Complex Matrix4x4::operator()(const std::size_t row,
                              const std::size_t col) const {
  return data[(row * K_COLS) + col];
}

Matrix4x4 Matrix4x4::operator*(const Matrix4x4& rhs) const {
  Matrix4x4 out{};
  multiply4x4(data, rhs.data, out.data);
  return out;
}

void Matrix4x4::premultiplyBy(const Matrix4x4& lhs) { *this = lhs * *this; }

Matrix4x4 Matrix4x4::operator*(const Complex& scalar) const {
  Matrix4x4 out = *this;
  out *= scalar;
  return out;
}

Matrix4x4& Matrix4x4::operator*=(const Complex& scalar) {
  for (Complex& entry : data) {
    entry *= scalar;
  }
  return *this;
}

Matrix4x4 Matrix4x4::adjoint() const {
  Matrix4x4 out{};
  adjointInto(data, out.data, K_ROWS);
  return out;
}

Matrix4x4 Matrix4x4::transpose() const {
  return fromElements(data[0], data[4], data[8], data[12], data[1], data[5],
                      data[9], data[13], data[2], data[6], data[10], data[14],
                      data[3], data[7], data[11], data[15]);
}

Complex Matrix4x4::trace() const {
  return data[0] + data[5] + data[10] + data[15];
}

Complex Matrix4x4::determinant() const {
  auto det3 = [](const Complex& m00, const Complex& m01, const Complex& m02,
                 const Complex& m10, const Complex& m11, const Complex& m12,
                 const Complex& m20, const Complex& m21, const Complex& m22) {
    return m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) +
           m02 * (m10 * m21 - m11 * m20);
  };
  return data[0] * det3(data[5], data[6], data[7], data[9], data[10], data[11],
                        data[13], data[14], data[15]) -
         data[1] * det3(data[4], data[6], data[7], data[8], data[10], data[11],
                        data[12], data[14], data[15]) +
         data[2] * det3(data[4], data[5], data[7], data[8], data[9], data[11],
                        data[12], data[13], data[15]) -
         data[3] * det3(data[4], data[5], data[6], data[8], data[9], data[10],
                        data[12], data[13], data[14]);
}

bool Matrix4x4::isIdentity(const double tol) const {
  return isIdentityEntries(data, K_ROWS, tol);
}

std::array<Complex, Matrix4x4::K_ROWS> Matrix4x4::diagonal() const {
  return {data[0], data[5], data[10], data[15]};
}

Matrix4x4 Matrix4x4::fromDiagonal(const ArrayRef<Complex> diagonalEntries) {
  assert(diagonalEntries.size() == K_ROWS &&
         "fromDiagonal requires exactly K_ROWS entries");
  Matrix4x4 out{};
  out.data[0] = diagonalEntries[0];
  out.data[5] = diagonalEntries[1];
  out.data[10] = diagonalEntries[2];
  out.data[15] = diagonalEntries[3];
  return out;
}

Matrix4x4 Matrix4x4::kron(const Matrix2x2& lhs, const Matrix2x2& rhs) {
  const auto& a = lhs.data;
  const auto& b = rhs.data;
  return fromElements(a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1],
                      a[0] * b[2], a[0] * b[3], a[1] * b[2], a[1] * b[3],
                      a[2] * b[0], a[2] * b[1], a[3] * b[0], a[3] * b[1],
                      a[2] * b[2], a[2] * b[3], a[3] * b[2], a[3] * b[3]);
}

std::array<Complex, Matrix4x4::K_ROWS>
Matrix4x4::column(const std::size_t col) const {
  assert(col < K_COLS);
  return {data[col], data[K_COLS + col], data[(2 * K_COLS) + col],
          data[(3 * K_COLS) + col]};
}

void Matrix4x4::setColumn(const std::size_t col,
                          const ArrayRef<Complex> values) {
  assert(col < K_COLS);
  assert(values.size() == K_ROWS &&
         "setColumn requires exactly K_ROWS entries");
  data[col] = values[0];
  data[K_COLS + col] = values[1];
  data[(2 * K_COLS) + col] = values[2];
  data[(3 * K_COLS) + col] = values[3];
}

ArrayRef<const Complex> Matrix4x4::row(const std::size_t row) const {
  assert(row < K_ROWS);
  return ArrayRef(data).slice(row * K_COLS, K_COLS);
}

void Matrix4x4::setRow(const std::size_t row, const ArrayRef<Complex> values) {
  assert(row < K_ROWS);
  assert(values.size() == K_COLS && "setRow requires exactly K_COLS entries");
  const std::size_t rowBase = row * K_COLS;
  data[rowBase + 0] = values[0];
  data[rowBase + 1] = values[1];
  data[rowBase + 2] = values[2];
  data[rowBase + 3] = values[3];
}

std::array<double, Matrix4x4::K_SIZE_AT_COMPILE_TIME>
Matrix4x4::realPart() const {
  return {data[0].real(),  data[1].real(),  data[2].real(),  data[3].real(),
          data[4].real(),  data[5].real(),  data[6].real(),  data[7].real(),
          data[8].real(),  data[9].real(),  data[10].real(), data[11].real(),
          data[12].real(), data[13].real(), data[14].real(), data[15].real()};
}

std::array<double, Matrix4x4::K_SIZE_AT_COMPILE_TIME>
Matrix4x4::imagPart() const {
  return {data[0].imag(),  data[1].imag(),  data[2].imag(),  data[3].imag(),
          data[4].imag(),  data[5].imag(),  data[6].imag(),  data[7].imag(),
          data[8].imag(),  data[9].imag(),  data[10].imag(), data[11].imag(),
          data[12].imag(), data[13].imag(), data[14].imag(), data[15].imag()};
}

bool Matrix4x4::isApprox(const Matrix4x4& other, const double tol) const {
  return entriesAreApprox(data, other.data, tol);
}

bool Matrix4x4::assignFrom(const DynamicMatrix& src) {
  return assignFromDynamicImpl<K_ROWS, K_SIZE_AT_COMPILE_TIME>(src, data);
}

DynamicMatrix Matrix4x4::embedInNqubit(const std::size_t numQubits,
                                       const std::size_t q0Index,
                                       const std::size_t q1Index) const {
  assert(q0Index < numQubits && q1Index < numQubits && q0Index != q1Index &&
         "Invalid qubit indices for two-qubit embed");
  if (numQubits == 2) {
    return DynamicMatrix(reorderForQubits(q0Index, q1Index));
  }
  const auto dim = checkedHilbertDim(numQubits);
  DynamicMatrix out(dim);
  const auto udim = static_cast<std::size_t>(dim);
  for (std::size_t row = 0; row < udim; ++row) {
    for (std::size_t col = 0; col < udim; ++col) {
      if (!otherQubitBitsMatch(row, col, numQubits, q0Index, q1Index)) {
        continue;
      }
      const std::size_t rowPair = (qubitBitAt(row, numQubits, q0Index) << 1) |
                                  qubitBitAt(row, numQubits, q1Index);
      const std::size_t colPair = (qubitBitAt(col, numQubits, q0Index) << 1) |
                                  qubitBitAt(col, numQubits, q1Index);
      out(static_cast<std::int64_t>(row), static_cast<std::int64_t>(col)) =
          (*this)(rowPair, colPair);
    }
  }
  return out;
}

Matrix4x4 Matrix4x4::reorderForQubits(const std::size_t q0Index,
                                      const std::size_t q1Index) const {
  if (q0Index == 0 && q1Index == 1) {
    return *this;
  }
  if (q0Index == 1 && q1Index == 0) {
    // Conjugate by SWAP: out[i, j] = matrix[pi(i), pi(j)] with pi swapping |01>
    // and |10> (basis indices 1 and 2).
    const auto& m = data;
    return fromElements(m[0], m[2], m[1], m[3], m[8], m[10], m[9], m[11], m[4],
                        m[6], m[5], m[7], m[12], m[14], m[13], m[15]);
  }
  llvm::reportFatalInternalError("Invalid qubit indices for two-qubit reorder");
}

SymmetricEigen4 Matrix4x4::symmetricEigen4() const {
  return symmetricEigen4(realPart());
}

// Adapted from John Burkardt's MIT-licensed EISPACK C port (`tred2` / `tql2`):
// https://people.sc.fsu.edu/~jburkardt/c_src/eispack/eispack.c
// Specialized to `n = 4`; input is row-major, accumulator `z` is column-major.

/// EISPACK `tred2` for `n = 4` (column-major `z[row + col*n]`).
static void symmetricTred24(const std::array<double, 16>& input,
                            std::array<double, 16>& z,
                            std::array<double, 4>& diag,
                            std::array<double, 4>& subdiag) {
  constexpr std::size_t n = 4;
  const auto zAt = [&z](const std::size_t row,
                        const std::size_t col) -> double& {
    return z[row + (col * n)];
  };
  double h = 0.0;

  for (std::size_t col = 0; col < n; ++col) {
    for (std::size_t row = col; row < n; ++row) {
      zAt(row, col) = input[(row * n) + col];
    }
    diag[col] = input[((n - 1) * n) + col];
  }

  for (int i = static_cast<int>(n) - 1; i >= 1; --i) {
    const auto ui = static_cast<std::size_t>(i);
    const std::size_t l = ui - 1;
    h = 0.0;
    double scale = 0.0;
    for (std::size_t k = 0; k <= l; ++k) {
      scale += std::abs(diag[k]);
    }
    if (scale == 0.0) {
      subdiag[ui] = diag[l];
      for (std::size_t j = 0; j <= l; ++j) {
        diag[j] = zAt(l, j);
        zAt(ui, j) = 0.0;
        zAt(j, ui) = 0.0;
      }
      diag[ui] = 0.0;
      continue;
    }
    for (std::size_t k = 0; k <= l; ++k) {
      diag[k] /= scale;
    }
    for (std::size_t k = 0; k <= l; ++k) {
      h += diag[k] * diag[k];
    }
    const double f = diag[l];
    const double g = -std::sqrt(h) * std::copysign(1.0, f);
    subdiag[ui] = scale * g;
    h -= f * g;
    diag[l] = f - g;

    for (std::size_t k = 0; k <= l; ++k) {
      subdiag[k] = 0.0;
    }
    for (std::size_t j = 0; j <= l; ++j) {
      const double fj = diag[j];
      zAt(j, ui) = fj;
      double gj = subdiag[j] + (zAt(j, j) * fj);
      for (std::size_t k = j + 1; k <= l; ++k) {
        gj += zAt(k, j) * diag[k];
        subdiag[k] += zAt(k, j) * fj;
      }
      subdiag[j] = gj;
    }
    double ff = 0.0;
    for (std::size_t k = 0; k <= l; ++k) {
      subdiag[k] /= h;
      ff += subdiag[k] * diag[k];
    }
    const double hh = 0.5 * ff / h;
    for (std::size_t k = 0; k <= l; ++k) {
      subdiag[k] -= hh * diag[k];
    }
    for (std::size_t j = 0; j <= l; ++j) {
      const double fj = diag[j];
      const double gj = subdiag[j];
      for (std::size_t k = j; k <= l; ++k) {
        zAt(k, j) -= (fj * subdiag[k]) + (gj * diag[k]);
      }
      diag[j] = zAt(l, j);
      zAt(ui, j) = 0.0;
    }
    diag[ui] = h;
  }

  for (std::size_t i = 1; i < n; ++i) {
    const std::size_t l = i - 1;
    zAt(n - 1, l) = zAt(l, l);
    zAt(l, l) = 1.0;
    h = diag[i];
    if (h != 0.0) {
      for (std::size_t k = 0; k <= l; ++k) {
        diag[k] = zAt(k, i) / h;
      }
      for (std::size_t j = 0; j <= l; ++j) {
        double g = 0.0;
        for (std::size_t k = 0; k <= l; ++k) {
          g += zAt(k, i) * zAt(k, j);
        }
        for (std::size_t k = 0; k <= l; ++k) {
          zAt(k, j) -= g * diag[k];
        }
      }
    }
    for (std::size_t k = 0; k <= l; ++k) {
      zAt(k, i) = 0.0;
    }
  }

  for (std::size_t j = 0; j < n; ++j) {
    diag[j] = zAt(n - 1, j);
  }
  for (std::size_t j = 0; j < n - 1; ++j) {
    zAt(n - 1, j) = 0.0;
  }
  zAt(n - 1, n - 1) = 1.0;
  subdiag[0] = 0.0;
}

/// EISPACK `tql2` for `n = 4` (column-major `z[row + col*n]`).
static void symmetricTql24(std::array<double, 4>& diag,
                           std::array<double, 4>& subdiag,
                           std::array<double, 16>& z) {
  constexpr std::size_t n = 4;
  const auto zAt = [&z](const std::size_t row,
                        const std::size_t col) -> double& {
    return z[row + (col * n)];
  };

  for (std::size_t i = 1; i < n; ++i) {
    subdiag[i - 1] = subdiag[i];
  }
  double f = 0.0;
  double tst1 = 0.0;
  subdiag[n - 1] = 0.0;

  for (std::size_t l = 0; l < n; ++l) {
    int j = 0;
    const double h = std::abs(diag[l]) + std::abs(subdiag[l]);
    tst1 = std::max(tst1, h);

    std::size_t m = l;
    for (; m < n; ++m) {
      const double tst2 = tst1 + std::abs(subdiag[m]);
      if (tst2 == tst1) {
        break;
      }
    }

    if (m != l) {
      while (true) {
        if (j == 30) {
          llvm::reportFatalInternalError("symmetricTql2_4: failed to converge");
        }
        ++j;

        const std::size_t l1 = l + 1;
        const std::size_t l2 = l1 + 1;
        const double g = diag[l];
        const double p = (diag[l1] - g) / (2.0 * subdiag[l]);
        const double r = std::hypot(p, 1.0);
        diag[l] = subdiag[l] / (p + std::copysign(std::abs(r), p));
        diag[l1] = subdiag[l] * (p + std::copysign(std::abs(r), p));
        const double dl1 = diag[l1];
        const double hh = g - diag[l];
        for (std::size_t i = l2; i < n; ++i) {
          diag[i] -= hh;
        }
        f += hh;

        double pv = diag[m];
        double c = 1.0;
        double c2 = c;
        const double el1 = subdiag[l1];
        double s = 0.0;
        double c3 = 1.0;
        double s2 = 0.0;
        const std::size_t mml = m - l;
        for (std::size_t ii = 1; ii <= mml; ++ii) {
          c3 = c2;
          c2 = c;
          s2 = s;
          const std::size_t i = m - ii;
          const double gi = c * subdiag[i];
          const double hi = c * pv;
          const double ri = std::hypot(pv, subdiag[i]);
          subdiag[i + 1] = s * ri;
          s = subdiag[i] / ri;
          c = pv / ri;
          pv = (c * diag[i]) - (s * gi);
          diag[i + 1] = hi + (s * ((c * gi) + (s * diag[i])));
          for (std::size_t k = 0; k < n; ++k) {
            const double zkI1 = zAt(k, i + 1);
            zAt(k, i + 1) = (s * zAt(k, i)) + (c * zkI1);
            zAt(k, i) = (c * zAt(k, i)) - (s * zkI1);
          }
        }
        pv = -s * s2 * c3 * el1 * subdiag[l] / dl1;
        subdiag[l] = s * pv;
        diag[l] = c * pv;
        const double tst2 = tst1 + std::abs(subdiag[l]);
        if (tst2 > tst1) {
          continue;
        }
        break;
      }
    }
    diag[l] += f;
  }

  for (std::size_t ii = 1; ii < n; ++ii) {
    const std::size_t i = ii - 1;
    std::size_t k = i;
    double p = diag[i];
    for (std::size_t j = ii; j < n; ++j) {
      if (diag[j] < p) {
        k = j;
        p = diag[j];
      }
    }
    if (k == i) {
      continue;
    }
    diag[k] = diag[i];
    diag[i] = p;
    for (std::size_t j = 0; j < n; ++j) {
      const double tmp = zAt(j, i);
      zAt(j, i) = zAt(j, k);
      zAt(j, k) = tmp;
    }
  }
}

SymmetricEigen4
Matrix4x4::symmetricEigen4(const std::array<double, 16>& symmetric) {
  constexpr std::size_t n = 4;

  SymmetricEigen4 result;
  std::array<double, 16> z{};
  std::array<double, 4> subdiag{};
  symmetricTred24(symmetric, z, result.eigenvalues, subdiag);
  symmetricTql24(result.eigenvalues, subdiag, z);

  for (std::size_t col = 0; col < n; ++col) {
    for (std::size_t row = 0; row < n; ++row) {
      result.eigenvectors(row, col) = z[row + (col * n)];
    }
  }
  return result;
}

namespace {
// Adapted from John Burkardt's MIT-licensed EISPACK C port (pythag, csroot) and
// NETLIB EISPACK Fortran (corth.f, comqr2.f, cdiv.f). Uses low=0, igh=n-1
// without CBAL balancing, matching EISPACK when balancing is skipped.

/// Row-major `ld x n` matrix view for EISPACK storage (`values[row + col *
/// ld]`).
class ComplexLdMatrix {
public:
  ComplexLdMatrix(MutableArrayRef<double> values, const int ld)
      : values_(values), ld_(ld) {}

  [[nodiscard]] static std::size_t rowMajorIndex(const int row, const int col,
                                                 const int ld) {
    return static_cast<std::size_t>(row) +
           (static_cast<std::size_t>(col) * static_cast<std::size_t>(ld));
  }

  [[nodiscard]] std::size_t linearIndex(const int row, const int col) const {
    return rowMajorIndex(row, col, ld_);
  }

  [[nodiscard]] double& at(const int row, const int col) {
    return values_[linearIndex(row, col)];
  }

  [[nodiscard]] const double& at(const int row, const int col) const {
    return values_[linearIndex(row, col)];
  }

private:
  MutableArrayRef<double> values_;
  int ld_;
};

[[nodiscard]] double complexPythag(const double a, const double b) {
  double p = std::max(std::abs(a), std::abs(b));
  if (p != 0.0) {
    double r = std::min(std::abs(a), std::abs(b)) / p;
    r = r * r;
    while (true) {
      const double t = 4.0 + r;
      if (t == 4.0) {
        break;
      }
      const double s = r / t;
      const double u = 1.0 + (2.0 * s);
      p = u * p;
      r = (s / u) * (s / u) * r;
    }
  }
  return p;
}

[[nodiscard]] std::pair<double, double> complexCsroot(const double xr,
                                                      const double xi) {
  const double tr = xr;
  const double ti = xi;
  const double s = std::sqrt(0.5 * (complexPythag(tr, ti) + std::abs(tr)));

  double yr = 0.0;
  double yi = 0.0;
  if (0.0 <= tr) {
    yr = s;
  }

  double sSign = s;
  if (ti < 0.0) {
    sSign = -s;
  }

  if (tr <= 0.0) {
    yi = sSign;
  }

  if (tr < 0.0) {
    yr = 0.5 * (ti / yi);
  } else if (0.0 < tr) {
    yi = 0.5 * (ti / yr);
  }
  return {yr, yi};
}

[[nodiscard]] std::pair<double, double> complexCdiv(const double ar,
                                                    const double ai,
                                                    const double br,
                                                    const double bi) {
  const double s = std::abs(br) + std::abs(bi);
  const double ars = ar / s;
  const double ais = ai / s;
  const double brs = br / s;
  const double bis = bi / s;
  const double denom = (brs * brs) + (bis * bis);
  return {(ars * brs + ais * bis) / denom, (ais * brs - ars * bis) / denom};
}

void complexCorth(const int nm, const int n, const int low, const int igh,
                  MutableArrayRef<double> arBuf, MutableArrayRef<double> aiBuf,
                  MutableArrayRef<double> ortrBuf,
                  MutableArrayRef<double> ortiBuf) {
  ComplexLdMatrix ar(arBuf, nm);
  ComplexLdMatrix ai(aiBuf, nm);
  const auto ortrAt = [&ortrBuf](const int index) -> double& {
    return ortrBuf[static_cast<std::size_t>(index)];
  };
  const auto ortiAt = [&ortiBuf](const int index) -> double& {
    return ortiBuf[static_cast<std::size_t>(index)];
  };

  const int kp1 = low + 1;
  const int la = igh - 1;
  if (la < kp1) {
    return;
  }

  for (int m = kp1; m <= la; ++m) {
    double h = 0.0;
    ortrAt(m) = 0.0;
    ortiAt(m) = 0.0;
    double scale = 0.0;
    const int subCol = m - 1;
    for (int i = m; i <= igh; ++i) {
      scale += std::abs(ar.at(i, subCol)) + std::abs(ai.at(i, subCol));
    }

    if (scale == 0.0) {
      continue;
    }

    const int mp = m + igh;
    for (int ii = m; ii <= igh; ++ii) {
      const int i = mp - ii;
      ortrAt(i) = ar.at(i, subCol) / scale;
      ortiAt(i) = ai.at(i, subCol) / scale;
      h += (ortrAt(i) * ortrAt(i)) + (ortiAt(i) * ortiAt(i));
    }

    double g = std::sqrt(h);
    const double f = complexPythag(ortrAt(m), ortiAt(m));
    if (f == 0.0) {
      ortrAt(m) = g;
      ar.at(m, subCol) = scale;
    } else {
      h += f * g;
      g /= f;
      ortrAt(m) = (1.0 + g) * ortrAt(m);
      ortiAt(m) = (1.0 + g) * ortiAt(m);
    }

    for (int j = m; j < n; ++j) {
      double fr = 0.0;
      double fi = 0.0;
      for (int ii = m; ii <= igh; ++ii) {
        const int i = mp - ii;
        fr += (ortrAt(i) * ar.at(i, j)) + (ortiAt(i) * ai.at(i, j));
        fi += (ortrAt(i) * ai.at(i, j)) - (ortiAt(i) * ar.at(i, j));
      }
      fr /= h;
      fi /= h;
      for (int i = m; i <= igh; ++i) {
        ar.at(i, j) -= (fr * ortrAt(i)) - (fi * ortiAt(i));
        ai.at(i, j) -= (fr * ortiAt(i)) + (fi * ortrAt(i));
      }
    }

    for (int i = 0; i <= igh; ++i) {
      double fr = 0.0;
      double fi = 0.0;
      for (int jj = m; jj <= igh; ++jj) {
        const int j = mp - jj;
        fr += (ortrAt(j) * ar.at(i, j)) - (ortiAt(j) * ai.at(i, j));
        fi += (ortrAt(j) * ai.at(i, j)) + (ortiAt(j) * ar.at(i, j));
      }
      fr /= h;
      fi /= h;
      for (int j = m; j <= igh; ++j) {
        ar.at(i, j) -= (fr * ortrAt(j)) + (fi * ortiAt(j));
        ai.at(i, j) += (fr * ortiAt(j)) - (fi * ortrAt(j));
      }
    }

    ortrAt(m) = scale * ortrAt(m);
    ortiAt(m) = scale * ortiAt(m);
    ar.at(m, subCol) = -g * ar.at(m, subCol);
    ai.at(m, subCol) = -g * ai.at(m, subCol);
  }
}

[[nodiscard]] int
complexComqr2(const int nm, const int n, const int low, const int igh,
              MutableArrayRef<double> ortrBuf, MutableArrayRef<double> ortiBuf,
              MutableArrayRef<double> hrBuf, MutableArrayRef<double> hiBuf,
              MutableArrayRef<double> wrBuf, MutableArrayRef<double> wiBuf,
              MutableArrayRef<double> zrBuf, MutableArrayRef<double> ziBuf) {
  ComplexLdMatrix hr(hrBuf, nm);
  ComplexLdMatrix hi(hiBuf, nm);
  ComplexLdMatrix zr(zrBuf, nm);
  ComplexLdMatrix zi(ziBuf, nm);
  const auto ortrAt = [&ortrBuf](const int index) -> double& {
    return ortrBuf[static_cast<std::size_t>(index)];
  };
  const auto ortiAt = [&ortiBuf](const int index) -> double& {
    return ortiBuf[static_cast<std::size_t>(index)];
  };
  const auto wrAt = [&wrBuf](const int index) -> double& {
    return wrBuf[static_cast<std::size_t>(index)];
  };
  const auto wiAt = [&wiBuf](const int index) -> double& {
    return wiBuf[static_cast<std::size_t>(index)];
  };

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      zr.at(i, j) = 0.0;
      zi.at(i, j) = 0.0;
    }
    zr.at(j, j) = 1.0;
  }

  const int iend = igh - low - 1;
  if (iend > 0) {
    for (int ii = 1; ii <= iend; ++ii) {
      const int i = igh - ii;
      if (ortrAt(i) == 0.0 && ortiAt(i) == 0.0) {
        continue;
      }
      if (hr.at(i, i - 1) == 0.0 && hi.at(i, i - 1) == 0.0) {
        continue;
      }
      const double ortNorm =
          (hr.at(i, i - 1) * ortrAt(i)) + (hi.at(i, i - 1) * ortiAt(i));
      const int ip1 = i + 1;
      for (int k = ip1; k <= igh; ++k) {
        ortrAt(k) = hr.at(k, i - 1);
        ortiAt(k) = hi.at(k, i - 1);
      }
      for (int j = i; j <= igh; ++j) {
        double sr = 0.0;
        double si = 0.0;
        for (int k = i; k <= igh; ++k) {
          sr += (ortrAt(k) * zr.at(k, j)) + (ortiAt(k) * zi.at(k, j));
          si += (ortrAt(k) * zi.at(k, j)) - (ortiAt(k) * zr.at(k, j));
        }
        sr /= ortNorm;
        si /= ortNorm;
        for (int k = i; k <= igh; ++k) {
          zr.at(k, j) += (sr * ortrAt(k)) - (si * ortiAt(k));
          zi.at(k, j) += (sr * ortiAt(k)) + (si * ortrAt(k));
        }
      }
    }
  }

  if (iend >= 0) {
    const int hessLow = low + 1;
    for (int i = hessLow; i <= igh; ++i) {
      const int ll = std::min(i + 1, igh);
      if (hi.at(i, i - 1) == 0.0) {
        continue;
      }
      const double hessNorm = complexPythag(hr.at(i, i - 1), hi.at(i, i - 1));
      const double yr = hr.at(i, i - 1) / hessNorm;
      const double yi = hi.at(i, i - 1) / hessNorm;
      hr.at(i, i - 1) = hessNorm;
      hi.at(i, i - 1) = 0.0;
      for (int j = i; j < n; ++j) {
        const double si = (yr * hi.at(i, j)) - (yi * hr.at(i, j));
        hr.at(i, j) = (yr * hr.at(i, j)) + (yi * hi.at(i, j));
        hi.at(i, j) = si;
      }
      for (int j = 0; j <= ll; ++j) {
        const double si = (yr * hi.at(j, i)) + (yi * hr.at(j, i));
        hr.at(j, i) = (yr * hr.at(j, i)) - (yi * hi.at(j, i));
        hi.at(j, i) = si;
      }
      for (int j = low; j <= igh; ++j) {
        const double si = (yr * zi.at(j, i)) + (yi * zr.at(j, i));
        zr.at(j, i) = (yr * zr.at(j, i)) - (yi * zi.at(j, i));
        zi.at(j, i) = si;
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    if (i >= low && i <= igh) {
      continue;
    }
    wrAt(i) = hr.at(i, i);
    wiAt(i) = hi.at(i, i);
  }

  int en = igh;
  double tr = 0.0;
  double ti = 0.0;
  int itn = 30 * n;

  while (en >= low) {
    int its = 0;
    const int enm1 = en - 1;

    while (true) {
      int l = low;
      for (int ll = low; ll <= en; ++ll) {
        l = en + low - ll;
        if (l == low) {
          break;
        }
        const double tst1Local = std::abs(hr.at((l - 1), l - 1)) +
                                 std::abs(hi.at((l - 1), l - 1)) +
                                 std::abs(hr.at(l, l)) + std::abs(hi.at(l, l));
        const double tst2Local = tst1Local + std::abs(hr.at(l, l - 1));
        if (tst2Local == tst1Local) {
          break;
        }
      }

      if (l == en) {
        break;
      }
      if (itn == 0) {
        return en;
      }

      double sr = hr.at(en, en);
      double si = hi.at(en, en);
      if (its == 10 || its == 20) {
        sr = std::abs(hr.at(en, enm1)) + std::abs(hr.at(enm1, en - 2));
        si = 0.0;
      } else {
        double xr = hr.at(enm1, en) * hr.at(en, enm1);
        double xi = hi.at(enm1, en) * hr.at(en, enm1);
        if (xr != 0.0 || xi != 0.0) {
          const double yr = (hr.at(enm1, enm1) - sr) / 2.0;
          const double yi = (hi.at(enm1, enm1) - si) / 2.0;
          auto [zzr, zzi] =
              complexCsroot((yr * yr) - (yi * yi) + xr, (2.0 * yr * yi) + xi);
          if ((yr * zzr) + (yi * zzi) < 0.0) {
            zzr = -zzr;
            zzi = -zzi;
          }
          std::tie(xr, xi) = complexCdiv(xr, xi, yr + zzr, yi + zzi);
          sr -= xr;
          si -= xi;
        }
      }

      for (int i = low; i <= en; ++i) {
        hr.at(i, i) -= sr;
        hi.at(i, i) -= si;
      }
      tr += sr;
      ti += si;
      ++its;
      --itn;

      {
        const int lp1 = l + 1;
        for (int i = lp1; i <= en; ++i) {
          sr = hr.at(i, i - 1);
          hr.at(i, i - 1) = 0.0;
          const double stepNorm = complexPythag(
              complexPythag(hr.at((i - 1), i - 1), hi.at((i - 1), i - 1)), sr);
          const double xr = hr.at((i - 1), i - 1) / stepNorm;
          wrAt(i - 1) = xr;
          const double xi = hi.at((i - 1), i - 1) / stepNorm;
          wiAt(i - 1) = xi;
          hr.at((i - 1), i - 1) = stepNorm;
          hi.at((i - 1), i - 1) = 0.0;
          hi.at(i, i - 1) = sr / stepNorm;

          for (int j = i; j < n; ++j) {
            const double yr = hr.at((i - 1), j);
            const double yi = hi.at((i - 1), j);
            const double zzr = hr.at(i, j);
            const double zzi = hi.at(i, j);
            hr.at((i - 1), j) = (xr * yr) + (xi * yi) + (hi.at(i, i - 1) * zzr);
            hi.at((i - 1), j) = (xr * yi) - (xi * yr) + (hi.at(i, i - 1) * zzi);
            hr.at(i, j) = (xr * zzr) - (xi * zzi) - (hi.at(i, i - 1) * yr);
            hi.at(i, j) = (xr * zzi) + (xi * zzr) - (hi.at(i, i - 1) * yi);
          }
        }
      }

      si = hi.at(en, en);
      if (si != 0.0) {
        const double stepNorm = complexPythag(hr.at(en, en), si);
        sr = hr.at(en, en) / stepNorm;
        si /= stepNorm;
        hr.at(en, en) = stepNorm;
        hi.at(en, en) = 0.0;
        if (en != n - 1) {
          const int ip1 = en + 1;
          for (int j = ip1; j < n; ++j) {
            const double yr = hr.at(en, j);
            const double yi = hi.at(en, j);
            hr.at(en, j) = (sr * yr) + (si * yi);
            hi.at(en, j) = (sr * yi) - (si * yr);
          }
        }
      }

      {
        const int lp1 = l + 1;
        for (int j = lp1; j <= en; ++j) {
          const double xr = wrAt(j - 1);
          const double xi = wiAt(j - 1);
          for (int i = 0; i <= j; ++i) {
            double yr = hr.at(i, j - 1);
            double yi = 0.0;
            const double zzr = hr.at(i, j);
            double zzi = hi.at(i, j);
            if (i == j) {
              yi = hi.at(i, j - 1);
              hi.at(i, j - 1) = (xr * yi) + (xi * yr) + (hi.at(j, j - 1) * zzi);
            }
            hr.at(i, j - 1) = (xr * yr) - (xi * yi) + (hi.at(j, j - 1) * zzr);
            hr.at(i, j) = (xr * zzr) + (xi * zzi) - (hi.at(j, j - 1) * yr);
            hi.at(i, j) = (xr * zzi) - (xi * zzr) - (hi.at(j, j - 1) * yi);
          }
          for (int i = low; i <= igh; ++i) {
            const double yr = zr.at(i, j - 1);
            const double yi = zi.at(i, j - 1);
            const double zzr = zr.at(i, j);
            const double zzi = zi.at(i, j);
            zr.at(i, j - 1) = (xr * yr) - (xi * yi) + (hi.at(j, j - 1) * zzr);
            zi.at(i, j - 1) = (xr * yi) + (xi * yr) + (hi.at(j, j - 1) * zzi);
            zr.at(i, j) = (xr * zzr) + (xi * zzi) - (hi.at(j, j - 1) * yr);
            zi.at(i, j) = (xr * zzi) - (xi * zzr) - (hi.at(j, j - 1) * yi);
          }
        }
      }

      if (si != 0.0) {
        for (int i = 0; i <= en; ++i) {
          const double yr = hr.at(i, en);
          const double yi = hi.at(i, en);
          hr.at(i, en) = (sr * yr) - (si * yi);
          hi.at(i, en) = (sr * yi) + (si * yr);
        }
        for (int i = low; i <= igh; ++i) {
          const double yr = zr.at(i, en);
          const double yi = zi.at(i, en);
          zr.at(i, en) = (sr * yr) - (si * yi);
          zi.at(i, en) = (sr * yi) + (si * yr);
        }
      }
    }

    hr.at(en, en) += tr;
    wrAt(en) = hr.at(en, en);
    hi.at(en, en) += ti;
    wiAt(en) = hi.at(en, en);
    en = enm1;
  }

  double norm = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      const double matrixNorm = std::abs(hr.at(i, j)) + std::abs(hi.at(i, j));
      norm = std::max(norm, matrixNorm);
    }
  }

  if (n != 1 && norm != 0.0) {
    for (int nn = 2; nn <= n; ++nn) {
      en = n + 2 - nn;
      double xr = wrAt(en);
      double xi = wiAt(en);
      hr.at(en, en) = 1.0;
      hi.at(en, en) = 0.0;
      const int enm1Back = en - 1;
      for (int ii = 1; ii <= enm1Back; ++ii) {
        const int i = en - ii;
        double zzr = 0.0;
        double zzi = 0.0;
        const int ip1 = i + 1;
        for (int j = ip1; j <= en; ++j) {
          zzr += (hr.at(i, j) * hr.at(j, en)) - (hi.at(i, j) * hi.at(j, en));
          zzi += (hr.at(i, j) * hi.at(j, en)) + (hi.at(i, j) * hr.at(j, en));
        }
        double yr = xr - wrAt(i);
        double yi = xi - wiAt(i);
        if (yr == 0.0 && yi == 0.0) {
          double tst1 = norm;
          yr = tst1;
          double tst2 = 0.0;
          do {
            yr = 0.01 * yr;
            tst2 = norm + yr;
          } while (tst2 <= tst1);
        }
        auto [divReal, divImag] = complexCdiv(zzr, zzi, yr, yi);
        hr.at(i, en) = divReal;
        hi.at(i, en) = divImag;
        const double trLocal = std::abs(hr.at(i, en)) + std::abs(hi.at(i, en));
        if (trLocal == 0.0) {
          continue;
        }
        double tst1 = trLocal;
        const double tst2 = tst1 + (1.0 / tst1);
        if (tst2 <= tst1) {
          continue;
        }
        for (int j = i; j <= en; ++j) {
          hr.at(j, en) /= trLocal;
          hi.at(j, en) /= trLocal;
        }
      }
    }

    for (int i = 0; i < n; ++i) {
      if (i >= low && i <= igh) {
        continue;
      }
      for (int j = i; j < n; ++j) {
        zr.at(i, j) = hr.at(i, j);
        zi.at(i, j) = hi.at(i, j);
      }
    }

    for (int jj = low; jj <= igh; ++jj) {
      const int j = igh + low - jj;
      const int m = std::min(j, igh);
      for (int i = low; i <= igh; ++i) {
        double zzr = 0.0;
        double zzi = 0.0;
        for (int k = low; k <= m; ++k) {
          zzr += (zr.at(i, k) * hr.at(k, j)) - (zi.at(i, k) * hi.at(k, j));
          zzi += (zr.at(i, k) * hi.at(k, j)) + (zi.at(i, k) * hr.at(k, j));
        }
        zr.at(i, j) = zzr;
        zi.at(i, j) = zzi;
      }
    }
  }

  return 0;
}

constexpr int K_COMPLEX_EIGEN4_SIZE = 4;
constexpr int K_COMPLEX_EIGEN4_LD = K_COMPLEX_EIGEN4_SIZE;
// EISPACK `comqr2` backsubstitution uses column/row index `n` in an `nm`-stride
// layout.
constexpr int K_COMPLEX_EIGEN4_MATRIX_STORAGE =
    (K_COMPLEX_EIGEN4_LD * K_COMPLEX_EIGEN4_SIZE) + K_COMPLEX_EIGEN4_SIZE + 1;
constexpr int K_COMPLEX_EIGEN4_EIGENVALUE_STORAGE = K_COMPLEX_EIGEN4_SIZE + 1;

[[nodiscard]] std::size_t complexMatrixStorageSize(const int ld,
                                                   const int cols) {
  return (static_cast<std::size_t>(ld) * static_cast<std::size_t>(cols)) +
         static_cast<std::size_t>(cols) + 1U;
}

[[nodiscard]] std::size_t complexEigenvalueStorageSize(const int eigenvalues) {
  return static_cast<std::size_t>(eigenvalues) + 1U;
}

[[nodiscard]] ComplexEigen4 buildComplexEigen4(
    const std::array<double, K_COMPLEX_EIGEN4_EIGENVALUE_STORAGE>& wr,
    const std::array<double, K_COMPLEX_EIGEN4_EIGENVALUE_STORAGE>& wi,
    const std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE>& zr,
    const std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE>& zi) {
  ComplexEigen4 result;
  for (int col = 0; col < K_COMPLEX_EIGEN4_SIZE; ++col) {
    result.eigenvalues[static_cast<std::size_t>(col)] = Complex(
        wr[static_cast<std::size_t>(col)], wi[static_cast<std::size_t>(col)]);
    double norm = 0.0;
    for (int row = 0; row < K_COMPLEX_EIGEN4_SIZE; ++row) {
      const std::size_t idx =
          ComplexLdMatrix::rowMajorIndex(row, col, K_COMPLEX_EIGEN4_SIZE);
      norm += (zr[idx] * zr[idx]) + (zi[idx] * zi[idx]);
    }
    norm = std::sqrt(norm);
    for (int row = 0; row < K_COMPLEX_EIGEN4_SIZE; ++row) {
      const std::size_t idx =
          ComplexLdMatrix::rowMajorIndex(row, col, K_COMPLEX_EIGEN4_SIZE);
      if (norm > MATRIX_TOLERANCE) {
        result.eigenvectors(static_cast<std::size_t>(row),
                            static_cast<std::size_t>(col)) =
            Complex(zr[idx] / norm, zi[idx] / norm);
      } else {
        result.eigenvectors(static_cast<std::size_t>(row),
                            static_cast<std::size_t>(col)) =
            Complex(zr[idx], zi[idx]);
      }
    }
  }
  return result;
}

void copyMatrix4x4ToEispack(
    const Matrix4x4& matrix,
    std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE>& ar,
    std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE>& ai) {
  for (std::size_t row = 0; row < Matrix4x4::K_ROWS; ++row) {
    for (std::size_t col = 0; col < Matrix4x4::K_COLS; ++col) {
      const Complex& value = matrix(row, col);
      const std::size_t idx = row + (col * Matrix4x4::K_ROWS);
      ar[idx] = std::real(value);
      ai[idx] = std::imag(value);
    }
  }
}

// Stack-specialized EISPACK `corth` / `comqr2` for `n = 4` (fixed-size arrays).
[[nodiscard]] std::optional<ComplexEigen4>
computeComplexEigen4(const Matrix4x4& matrix) {
  constexpr int n = K_COMPLEX_EIGEN4_SIZE;
  constexpr int nm = n;
  constexpr int low = 0;
  constexpr int igh = n - 1;

  std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE> ar{};
  std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE> ai{};
  copyMatrix4x4ToEispack(matrix, ar, ai);

  std::array<double, 4> ortr{};
  std::array<double, 4> orti{};
  std::array<double, K_COMPLEX_EIGEN4_EIGENVALUE_STORAGE> wr{};
  std::array<double, K_COMPLEX_EIGEN4_EIGENVALUE_STORAGE> wi{};
  std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE> zr{};
  std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE> zi{};

  complexCorth(nm, n, low, igh, ar, ai, ortr, orti);
  const int ierr =
      complexComqr2(nm, n, low, igh, ortr, orti, ar, ai, wr, wi, zr, zi);
  if (ierr != 0) {
    return std::nullopt;
  }
  return buildComplexEigen4(wr, wi, zr, zi);
}

void normalizeVector(SmallVector<Complex>& vector) {
  const double norm =
      std::sqrt(std::accumulate(vector.begin(), vector.end(), 0.0,
                                [](const double sum, const Complex& value) {
                                  return sum + std::norm(value);
                                }));
  if (norm <= MATRIX_TOLERANCE) {
    return;
  }
  for (Complex& value : vector) {
    value /= norm;
  }
}

[[nodiscard]] std::optional<ComplexEigen>
complexEigen1x1(const DynamicMatrix& matrix) {
  ComplexEigen result;
  result.eigenvalues.push_back(matrix(0, 0));
  result.eigenvectors = DynamicMatrix(1);
  result.eigenvectors(0, 0) = 1.0;
  return result;
}

[[nodiscard]] ComplexEigen2 computeComplexEigen2(const Matrix2x2& matrix) {
  const Complex a = matrix(0, 0);
  const Complex b = matrix(0, 1);
  const Complex c = matrix(1, 0);
  const Complex d = matrix(1, 1);
  const Complex trace = a + d;
  const Complex determinant = a * d - b * c;
  const Complex discriminant = std::sqrt(trace * trace - 4.0 * determinant);
  const Complex lambda0 = (trace + discriminant) * 0.5;
  const Complex lambda1 = (trace - discriminant) * 0.5;

  auto eigenvectorFor = [&](const Complex& lambda) -> SmallVector<Complex> {
    SmallVector<Complex> vector(2, Complex{0.0, 0.0});
    if (std::abs(b) <= MATRIX_TOLERANCE && std::abs(c) <= MATRIX_TOLERANCE) {
      if (std::abs(lambda - a) <= MATRIX_TOLERANCE) {
        vector[0] = 1.0;
        vector[1] = 0.0;
      } else {
        vector[0] = 0.0;
        vector[1] = 1.0;
      }
    } else if (std::abs(b) > MATRIX_TOLERANCE) {
      vector[0] = b;
      vector[1] = lambda - a;
    } else {
      vector[0] = lambda - d;
      vector[1] = c;
    }
    normalizeVector(vector);
    return vector;
  };

  const SmallVector<Complex> vector0 = eigenvectorFor(lambda0);
  const SmallVector<Complex> vector1 = eigenvectorFor(lambda1);

  ComplexEigen2 result;
  result.eigenvalues = {lambda0, lambda1};
  result.eigenvectors(0, 0) = vector0[0];
  result.eigenvectors(1, 0) = vector0[1];
  result.eigenvectors(0, 1) = vector1[0];
  result.eigenvectors(1, 1) = vector1[1];
  return result;
}

[[nodiscard]] std::optional<ComplexEigen>
toComplexEigen(const ComplexEigen2& eigen2) {
  ComplexEigen result;
  result.eigenvalues.assign(eigen2.eigenvalues.begin(),
                            eigen2.eigenvalues.end());
  result.eigenvectors = DynamicMatrix(eigen2.eigenvectors);
  return result;
}

[[nodiscard]] std::optional<ComplexEigen>
toComplexEigen(const ComplexEigen4& eigen4) {
  ComplexEigen result;
  result.eigenvalues.assign(eigen4.eigenvalues.begin(),
                            eigen4.eigenvalues.end());
  result.eigenvectors = DynamicMatrix(eigen4.eigenvectors);
  return result;
}

[[nodiscard]] std::optional<ComplexEigen>
computeComplexEigen(const DynamicMatrix& matrix) {
  const std::int64_t dim = matrix.rows();
  assert(dim == matrix.cols() && dim >= 3 && dim != 4);
  const int n = static_cast<int>(dim);
  const int nm = n;
  const int low = 0;
  const int igh = n - 1;

  const std::size_t matrixStorage = complexMatrixStorageSize(nm, n);
  SmallVector<double> ar(matrixStorage);
  SmallVector<double> ai(matrixStorage);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      const Complex value = matrix(row, col);
      const std::size_t idx = ComplexLdMatrix::rowMajorIndex(row, col, n);
      ar[idx] = std::real(value);
      ai[idx] = std::imag(value);
    }
  }

  SmallVector<double> ortr(static_cast<std::size_t>(n));
  SmallVector<double> orti(static_cast<std::size_t>(n));
  SmallVector<double> wr(complexEigenvalueStorageSize(n));
  SmallVector<double> wi(complexEigenvalueStorageSize(n));
  SmallVector<double> zr(matrixStorage);
  SmallVector<double> zi(matrixStorage);

  complexCorth(nm, n, low, igh, ar, ai, ortr, orti);
  const int ierr =
      complexComqr2(nm, n, low, igh, ortr, orti, ar, ai, wr, wi, zr, zi);
  if (ierr != 0) {
    return std::nullopt;
  }

  ComplexEigen result;
  result.eigenvalues.reserve(static_cast<std::size_t>(n));
  result.eigenvectors = DynamicMatrix(dim);
  for (int col = 0; col < n; ++col) {
    result.eigenvalues.emplace_back(wr[static_cast<std::size_t>(col)],
                                    wi[static_cast<std::size_t>(col)]);
    double norm = 0.0;
    for (int row = 0; row < n; ++row) {
      const std::size_t idx = ComplexLdMatrix::rowMajorIndex(row, col, n);
      norm += (zr[idx] * zr[idx]) + (zi[idx] * zi[idx]);
    }
    norm = std::sqrt(norm);
    for (int row = 0; row < n; ++row) {
      const std::size_t idx = ComplexLdMatrix::rowMajorIndex(row, col, n);
      if (norm > MATRIX_TOLERANCE) {
        result.eigenvectors(row, col) = Complex(zr[idx] / norm, zi[idx] / norm);
      } else {
        result.eigenvectors(row, col) = Complex(zr[idx], zi[idx]);
      }
    }
  }
  return result;
}

} // namespace

ComplexEigen2 Matrix2x2::complexEigen() const {
  return computeComplexEigen2(*this);
}

std::optional<ComplexEigen4> Matrix4x4::complexEigen() const {
  return computeComplexEigen4(*this);
}

struct DynamicMatrix::Impl {
  std::int64_t dim = 0;
  SmallVector<Complex> data;
};

DynamicMatrix::DynamicMatrix() : impl_(std::make_unique<Impl>()) {}

DynamicMatrix::DynamicMatrix(const std::int64_t dim)
    : impl_(std::make_unique<Impl>()) {
  impl_->dim = dim;
  impl_->data.assign(checkedStorageSize(dim), Complex{});
}

DynamicMatrix::DynamicMatrix(const Matrix2x2& src)
    : impl_(std::make_unique<Impl>()) {
  assignFrom(src);
}

DynamicMatrix::DynamicMatrix(const Matrix4x4& src)
    : impl_(std::make_unique<Impl>()) {
  assignFrom(src);
}

DynamicMatrix::DynamicMatrix(const DynamicMatrix& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

DynamicMatrix::DynamicMatrix(DynamicMatrix&& other) noexcept = default;

DynamicMatrix& DynamicMatrix::operator=(const DynamicMatrix& other) {
  if (this != &other) {
    *impl_ = *other.impl_;
  }
  return *this;
}

DynamicMatrix&
DynamicMatrix::operator=(DynamicMatrix&& other) noexcept = default;

DynamicMatrix::~DynamicMatrix() = default;

std::int64_t DynamicMatrix::rows() const { return impl_->dim; }

std::int64_t DynamicMatrix::cols() const { return impl_->dim; }

DynamicMatrix DynamicMatrix::identity(const std::int64_t dim) {
  DynamicMatrix matrix(dim);
  const auto udim = checkedDim(dim);
  for (std::size_t i = 0; i < udim; ++i) {
    matrix.impl_->data[(i * udim) + i] = 1.0;
  }
  return matrix;
}

DynamicMatrix DynamicMatrix::fromAdjoint(const Matrix2x2& src) {
  return DynamicMatrix(src.adjoint());
}

Complex& DynamicMatrix::operator()(const std::int64_t row,
                                   const std::int64_t col) {
  return impl_->data[static_cast<std::size_t>((row * impl_->dim) + col)];
}

Complex DynamicMatrix::operator()(const std::int64_t row,
                                  const std::int64_t col) const {
  return impl_->data[static_cast<std::size_t>((row * impl_->dim) + col)];
}

void DynamicMatrix::setBottomRightCorner(const Matrix2x2& block) {
  copyBottomRightCorner(impl_->dim, impl_->data,
                        static_cast<std::int64_t>(Matrix2x2::K_ROWS),
                        block.data);
}

void DynamicMatrix::setBottomRightCorner(const Matrix4x4& block) {
  copyBottomRightCorner(impl_->dim, impl_->data,
                        static_cast<std::int64_t>(Matrix4x4::K_ROWS),
                        block.data);
}

void DynamicMatrix::setBottomRightCorner(const DynamicMatrix& block) {
  copyBottomRightCorner(impl_->dim, impl_->data, block.impl_->dim,
                        block.impl_->data);
}

DynamicMatrix DynamicMatrix::adjoint() const {
  DynamicMatrix out(impl_->dim);
  adjointInto(impl_->data, out.impl_->data, checkedDim(impl_->dim));
  return out;
}

void DynamicMatrix::assignFrom(const Matrix1x1& src) {
  impl_->dim = 1;
  impl_->data.assign({src.value});
}

void DynamicMatrix::assignFrom(const Matrix2x2& src) {
  assignFixedImpl<Matrix2x2::K_ROWS, Matrix2x2::K_SIZE_AT_COMPILE_TIME>(
      impl_->dim, impl_->data, src.data);
}

void DynamicMatrix::assignFrom(const Matrix4x4& src) {
  assignFixedImpl<Matrix4x4::K_ROWS, Matrix4x4::K_SIZE_AT_COMPILE_TIME>(
      impl_->dim, impl_->data, src.data);
}

void DynamicMatrix::assignFrom(const DynamicMatrix& src) {
  *impl_ = *src.impl_;
}

bool DynamicMatrix::isApprox(const Matrix1x1& other, const double tol) const {
  if (impl_->dim != 1) {
    return false;
  }
  return entryIsApprox(impl_->data[0], other.value, tol);
}

bool DynamicMatrix::isApprox(const Matrix2x2& other, const double tol) const {
  return isApproxFixedImpl<Matrix2x2::K_ROWS,
                           Matrix2x2::K_SIZE_AT_COMPILE_TIME>(
      impl_->dim, impl_->data, other.data, tol);
}

bool DynamicMatrix::isApprox(const Matrix4x4& other, const double tol) const {
  return isApproxFixedImpl<Matrix4x4::K_ROWS,
                           Matrix4x4::K_SIZE_AT_COMPILE_TIME>(
      impl_->dim, impl_->data, other.data, tol);
}

bool DynamicMatrix::isApprox(const DynamicMatrix& other,
                             const double tol) const {
  return entriesAreApprox(impl_->data, other.impl_->data, tol);
}

Complex DynamicMatrix::trace() const {
  Complex sum{0.0, 0.0};
  const auto udim = checkedDim(impl_->dim);
  for (std::size_t i = 0; i < udim; ++i) {
    sum += impl_->data[(i * udim) + i];
  }
  return sum;
}

DynamicMatrix DynamicMatrix::operator*(const DynamicMatrix& rhs) const {
  assert(impl_->dim == rhs.impl_->dim &&
         "DynamicMatrix multiply requires matching dimensions");
  DynamicMatrix out(impl_->dim);
  if (std::cmp_equal(impl_->dim, Matrix2x2::K_ROWS)) {
    multiply2x2(impl_->data, rhs.impl_->data, out.impl_->data);
    return out;
  }
  if (std::cmp_equal(impl_->dim, Matrix4x4::K_ROWS)) {
    multiply4x4(impl_->data, rhs.impl_->data, out.impl_->data);
    return out;
  }

  const auto udim = checkedDim(impl_->dim);
  for (std::size_t row = 0; row < udim; ++row) {
    for (std::size_t col = 0; col < udim; ++col) {
      Complex sum{0.0, 0.0};
      for (std::size_t k = 0; k < udim; ++k) {
        sum +=
            impl_->data[(row * udim) + k] * rhs.impl_->data[(k * udim) + col];
      }
      out.impl_->data[(row * udim) + col] = sum;
    }
  }
  return out;
}

DynamicMatrix DynamicMatrix::operator*(const Complex& scalar) const {
  DynamicMatrix out(impl_->dim);
  for (std::size_t i = 0; i < impl_->data.size(); ++i) {
    out.impl_->data[i] = impl_->data[i] * scalar;
  }
  return out;
}

DynamicMatrix& DynamicMatrix::operator*=(const Complex& scalar) {
  for (Complex& entry : impl_->data) {
    entry *= scalar;
  }
  return *this;
}

bool DynamicMatrix::isIdentity(const double tol) const {
  return isIdentityEntries(impl_->data, checkedDim(impl_->dim), tol);
}

Matrix2x2 operator*(const Complex& scalar, const Matrix2x2& matrix) {
  return matrix * scalar;
}

Matrix4x4 operator*(const Complex& scalar, const Matrix4x4& matrix) {
  return matrix * scalar;
}

DynamicMatrix operator*(const Complex& scalar, const DynamicMatrix& matrix) {
  return matrix * scalar;
}

std::optional<ComplexEigen> DynamicMatrix::complexEigen() const {
  const std::size_t dim = checkedDim(impl_->dim);
  if (dim == 0) {
    return std::nullopt;
  }
  if (dim == 1) {
    return complexEigen1x1(*this);
  }
  if (dim == 2) {
    Matrix2x2 fixed;
    if (!fixed.assignFrom(*this)) {
      return std::nullopt;
    }
    return toComplexEigen(fixed.complexEigen());
  }
  if (dim == 4) {
    Matrix4x4 fixed;
    if (!fixed.assignFrom(*this)) {
      return std::nullopt;
    }
    const std::optional<ComplexEigen4> eigen4 = fixed.complexEigen();
    if (!eigen4) {
      return std::nullopt;
    }
    return toComplexEigen(*eigen4);
  }
  return computeComplexEigen(*this);
}

} // namespace mlir::qco
