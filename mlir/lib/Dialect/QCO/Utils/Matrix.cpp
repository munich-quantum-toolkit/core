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
#include <llvm/ADT/SmallVector.h>
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
#include <optional>
#include <tuple>
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
                        const size_t dim) {
  for (size_t row = 0; row < dim; ++row) {
    for (size_t col = 0; col < dim; ++col) {
      out[(row * dim) + col] = std::conj(in[(col * dim) + row]);
    }
  }
}

template <size_t Dim, size_t Size>
static void assignFixedImpl(int64_t& dim, SmallVector<Complex>& data,
                            const std::array<Complex, Size>& src) {
  dim = static_cast<int64_t>(Dim);
  data.assign(src.begin(), src.end());
}

template <size_t Dim, size_t Size>
[[nodiscard]] static bool
isApproxFixedImpl(const int64_t dim, ArrayRef<Complex> data,
                  const std::array<Complex, Size>& other, const double tol) {
  if (std::cmp_not_equal(dim, Dim)) {
    return false;
  }
  return entriesAreApprox(data, other, tol);
}

template <size_t Dim, size_t Size>
[[nodiscard]] static bool
assignFromDynamicImpl(const DynamicMatrix& src,
                      std::array<Complex, Size>& dst) {
  if (src.rows() != static_cast<int64_t>(Dim) ||
      src.cols() != static_cast<int64_t>(Dim)) {
    return false;
  }
  for (size_t row = 0; row < Dim; ++row) {
    for (size_t col = 0; col < Dim; ++col) {
      dst[(row * Dim) + col] =
          src(static_cast<int64_t>(row), static_cast<int64_t>(col));
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
  for (size_t row = 0; row < Matrix4x4::K_ROWS; ++row) {
    const size_t rowBase = row * Matrix4x4::K_COLS;
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

/// Left-applies a 2x2 gate to a row pair (`out = gate * [a; b]`).
static void apply2x2LeftToRowPair(const ArrayRef<Complex> gate, Complex& a,
                                  Complex& b) {
  assert(gate.size() == Matrix2x2::K_SIZE_AT_COMPILE_TIME);
  const Complex newA = gate[0] * a + gate[1] * b;
  const Complex newB = gate[2] * a + gate[3] * b;
  a = newA;
  b = newB;
}

/// Left-applies a 4x4 gate to a column 4-vector (`[a; b; c; d] = gate *
/// [a; b; c; d]`).
static void apply4x4LeftToColumn(const ArrayRef<Complex> gate, Complex& a,
                                 Complex& b, Complex& c, Complex& d) {
  assert(gate.size() == Matrix4x4::K_SIZE_AT_COMPILE_TIME);
  const Complex newA = gate[0] * a + gate[1] * b + gate[2] * c + gate[3] * d;
  const Complex newB = gate[4] * a + gate[5] * b + gate[6] * c + gate[7] * d;
  const Complex newC = gate[8] * a + gate[9] * b + gate[10] * c + gate[11] * d;
  const Complex newD =
      gate[12] * a + gate[13] * b + gate[14] * c + gate[15] * d;
  a = newA;
  b = newB;
  c = newC;
  d = newD;
}

/// Returns true if @p data is approximately the @p dim x @p dim identity
/// matrix.
[[nodiscard]] static bool
isIdentityEntries(ArrayRef<Complex> data, const size_t dim, const double tol) {
  assert(data.size() >= dim * dim);
  for (size_t row = 0; row < dim; ++row) {
    for (size_t col = 0; col < dim; ++col) {
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
[[nodiscard]] static size_t checkedDim(const int64_t dim) {
  assert(dim >= 0 && "DynamicMatrix dimension must be non-negative");
  const auto udim = static_cast<size_t>(dim);
  assert((udim == 0 || udim <= std::numeric_limits<size_t>::max() / udim) &&
         "DynamicMatrix dimension is too large to allocate storage");
  return udim;
}

/// Returns the flat row-major index for `(row, col)` after bounds checking.
[[nodiscard]] static size_t checkedFlatIndex(const size_t row, const size_t col,
                                             const size_t dim) {
  assert(row < dim && col < dim && "matrix index out of bounds");
  return (row * dim) + col;
}

[[nodiscard]] static int64_t
checkedFlatIndex(const int64_t row, const int64_t col, const int64_t dim) {
  assert(row >= 0 && col >= 0 && row < dim && col < dim &&
         "matrix index out of bounds");
  return (row * dim) + col;
}

[[nodiscard]] static size_t checkedStorageSize(const int64_t dim) {
  const auto udim = checkedDim(dim);
  return udim * udim;
}

/// Returns `2^numQubits` as `int64_t` after checking it fits.
[[nodiscard]] static int64_t checkedHilbertDim(const size_t numQubits) {
  assert(numQubits < std::numeric_limits<int64_t>::digits &&
         "Hilbert-space dimension must fit in int64_t");
  return static_cast<int64_t>(uint64_t{1} << numQubits);
}

static void validateCornerDims(const int64_t matrixDim,
                               const int64_t blockDim) {
  assert(matrixDim >= 0 && blockDim >= 0 && blockDim <= matrixDim &&
         "block must fit in the bottom-right corner of the matrix");
  std::ignore = checkedDim(matrixDim);
}

/// Copies @p blockData into the bottom-right @p blockDim x @p blockDim corner.
static void copyBottomRightCorner(const int64_t matrixDim,
                                  MutableArrayRef<Complex> matrixData,
                                  const int64_t blockDim,
                                  ArrayRef<Complex> blockData) {
  validateCornerDims(matrixDim, blockDim);
  assert(matrixData.size() >= checkedStorageSize(matrixDim));
  assert(blockData.size() >= checkedStorageSize(blockDim));
  const int64_t offset = matrixDim - blockDim;
  for (int64_t row = 0; row < blockDim; ++row) {
    for (int64_t col = 0; col < blockDim; ++col) {
      matrixData[static_cast<size_t>(((offset + row) * matrixDim) + offset +
                                     col)] =
          blockData[static_cast<size_t>((row * blockDim) + col)];
    }
  }
}

/**
 * @brief Returns the @p qubitIndex bit of a computational-basis label.
 *
 * Qubit 0 is the MSB of @p stateIndex, matching @ref Matrix4x4::kron and
 * @ref Matrix2x2::embedInNqubit.
 */
[[nodiscard]] static size_t qubitBitAt(const size_t stateIndex,
                                       const size_t numQubits,
                                       const size_t qubitIndex) {
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
[[nodiscard]] static bool
otherQubitBitsMatch(const size_t row, const size_t col, const size_t numQubits,
                    const size_t skipA, const size_t skipB) {
  for (size_t q = 0; q < numQubits; ++q) {
    if (q == skipA || q == skipB) {
      continue;
    }
    if (qubitBitAt(row, numQubits, q) != qubitBitAt(col, numQubits, q)) {
      return false;
    }
  }
  return true;
}

Complex& Matrix1x1::operator()(const size_t row, const size_t col) {
  assert(row == 0 && col == 0 && "matrix index out of bounds");
  return value;
}

Complex Matrix1x1::operator()(const size_t row, const size_t col) const {
  assert(row == 0 && col == 0 && "matrix index out of bounds");
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

Complex& Matrix2x2::operator()(const size_t row, const size_t col) {
  return data[checkedFlatIndex(row, col, K_COLS)];
}

Complex Matrix2x2::operator()(const size_t row, const size_t col) const {
  return data[checkedFlatIndex(row, col, K_COLS)];
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

DynamicMatrix Matrix2x2::embedInNqubit(const size_t numQubits,
                                       const size_t qubitIndex) const {
  assert(qubitIndex < numQubits &&
         "Invalid qubit index for single-qubit embed");
  if (numQubits == 2) {
    return DynamicMatrix(embedInTwoQubit(qubitIndex));
  }
  const auto dim = checkedHilbertDim(numQubits);
  DynamicMatrix out(dim);
  const auto udim = static_cast<size_t>(dim);
  for (size_t row = 0; row < udim; ++row) {
    for (size_t col = 0; col < udim; ++col) {
      if (!otherQubitBitsMatch(row, col, numQubits, qubitIndex, numQubits)) {
        continue;
      }
      const size_t rowBit = qubitBitAt(row, numQubits, qubitIndex);
      const size_t colBit = qubitBitAt(col, numQubits, qubitIndex);
      out(static_cast<int64_t>(row), static_cast<int64_t>(col)) =
          (*this)(rowBit, colBit);
    }
  }
  return out;
}

Matrix4x4 Matrix2x2::embedInTwoQubit(const size_t qubitIndex) const {
  if (qubitIndex == 0) {
    return Matrix4x4::kron(*this, Matrix2x2::identity());
  }
  if (qubitIndex == 1) {
    return Matrix4x4::kron(Matrix2x2::identity(), *this);
  }
  llvm::reportFatalInternalError("Invalid qubit index for single-qubit embed");
}

Complex& Matrix4x4::operator()(const size_t row, const size_t col) {
  return data[checkedFlatIndex(row, col, K_COLS)];
}

Complex Matrix4x4::operator()(const size_t row, const size_t col) const {
  return data[checkedFlatIndex(row, col, K_COLS)];
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

Matrix4x4 Matrix4x4::kron(const Matrix2x2& lhs, const Matrix2x2& rhs) {
  const auto& a = lhs.data;
  const auto& b = rhs.data;
  return fromElements(a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1],
                      a[0] * b[2], a[0] * b[3], a[1] * b[2], a[1] * b[3],
                      a[2] * b[0], a[2] * b[1], a[3] * b[0], a[3] * b[1],
                      a[2] * b[2], a[2] * b[3], a[3] * b[2], a[3] * b[3]);
}

std::array<Complex, Matrix4x4::K_ROWS>
Matrix4x4::column(const size_t col) const {
  assert(col < K_COLS && "matrix index out of bounds");
  return {data[col], data[K_COLS + col], data[(2 * K_COLS) + col],
          data[(3 * K_COLS) + col]};
}

void Matrix4x4::setColumn(const size_t col, const ArrayRef<Complex> values) {
  assert(col < K_COLS && "matrix index out of bounds");
  assert(values.size() == K_ROWS &&
         "setColumn requires exactly K_ROWS entries");
  data[col] = values[0];
  data[K_COLS + col] = values[1];
  data[(2 * K_COLS) + col] = values[2];
  data[(3 * K_COLS) + col] = values[3];
}

ArrayRef<const Complex> Matrix4x4::row(const size_t row) const {
  assert(row < K_ROWS && "matrix index out of bounds");
  return ArrayRef(data).slice(row * K_COLS, K_COLS);
}

void Matrix4x4::setRow(const size_t row, const ArrayRef<Complex> values) {
  assert(row < K_ROWS && "matrix index out of bounds");
  assert(values.size() == K_COLS && "setRow requires exactly K_COLS entries");
  const size_t rowBase = row * K_COLS;
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

DynamicMatrix Matrix4x4::embedInNqubit(const size_t numQubits,
                                       const size_t q0Index,
                                       const size_t q1Index) const {
  assert(q0Index < numQubits && q1Index < numQubits && q0Index != q1Index &&
         "Invalid qubit indices for two-qubit embed");
  if (numQubits == 2) {
    return DynamicMatrix(reorderForQubits(q0Index, q1Index));
  }
  const auto dim = checkedHilbertDim(numQubits);
  DynamicMatrix out(dim);
  const auto udim = static_cast<size_t>(dim);
  for (size_t row = 0; row < udim; ++row) {
    for (size_t col = 0; col < udim; ++col) {
      if (!otherQubitBitsMatch(row, col, numQubits, q0Index, q1Index)) {
        continue;
      }
      const size_t rowPair = (qubitBitAt(row, numQubits, q0Index) << 1) |
                             qubitBitAt(row, numQubits, q1Index);
      const size_t colPair = (qubitBitAt(col, numQubits, q0Index) << 1) |
                             qubitBitAt(col, numQubits, q1Index);
      out(static_cast<int64_t>(row), static_cast<int64_t>(col)) =
          (*this)(rowPair, colPair);
    }
  }
  return out;
}

Matrix4x4 Matrix4x4::reorderForQubits(const size_t q0Index,
                                      const size_t q1Index) const {
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

// Eigensolver implementations (EISPACK-derived).
// Adapted from John Burkardt's MIT-licensed EISPACK C port (`tred2` / `tql2`):
// https://people.sc.fsu.edu/~jburkardt/c_src/eispack/eispack.c
// Original Fortran: https://netlib.org/eispack/tred2.f,
// https://netlib.org/eispack/tql2.f
// Specialized to `n = 4`; input is row-major, accumulator `z` is column-major.

/// EISPACK `tred2` for `n = 4` (column-major `z[row + col*n]`).
static void symmetricTred24(ArrayRef<double> input, std::array<double, 16>& z,
                            std::array<double, 4>& diag,
                            std::array<double, 4>& subdiag) {
  constexpr size_t n = 4;
  const auto zAt = [&z](const size_t row, const size_t col) -> double& {
    return z[row + (col * n)];
  };
  double h = 0.0;

  for (size_t col = 0; col < n; ++col) {
    for (size_t row = col; row < n; ++row) {
      zAt(row, col) = input[(row * n) + col];
    }
    diag[col] = input[((n - 1) * n) + col];
  }

  for (int i = static_cast<int>(n) - 1; i >= 1; --i) {
    const auto ui = static_cast<size_t>(i);
    const size_t l = ui - 1;
    h = 0.0;
    double scale = 0.0;
    for (size_t k = 0; k <= l; ++k) {
      scale += std::abs(diag[k]);
    }
    if (scale == 0.0) {
      subdiag[ui] = diag[l];
      for (size_t j = 0; j <= l; ++j) {
        diag[j] = zAt(l, j);
        zAt(ui, j) = 0.0;
        zAt(j, ui) = 0.0;
      }
      diag[ui] = 0.0;
      continue;
    }
    for (size_t k = 0; k <= l; ++k) {
      diag[k] /= scale;
    }
    for (size_t k = 0; k <= l; ++k) {
      h += diag[k] * diag[k];
    }
    const double f = diag[l];
    const double g = -std::sqrt(h) * std::copysign(1.0, f);
    subdiag[ui] = scale * g;
    h -= f * g;
    diag[l] = f - g;

    for (size_t k = 0; k <= l; ++k) {
      subdiag[k] = 0.0;
    }
    for (size_t j = 0; j <= l; ++j) {
      const double fj = diag[j];
      zAt(j, ui) = fj;
      double gj = subdiag[j] + (zAt(j, j) * fj);
      for (size_t k = j + 1; k <= l; ++k) {
        gj += zAt(k, j) * diag[k];
        subdiag[k] += zAt(k, j) * fj;
      }
      subdiag[j] = gj;
    }
    double ff = 0.0;
    for (size_t k = 0; k <= l; ++k) {
      subdiag[k] /= h;
      ff += subdiag[k] * diag[k];
    }
    const double hh = 0.5 * ff / h;
    for (size_t k = 0; k <= l; ++k) {
      subdiag[k] -= hh * diag[k];
    }
    for (size_t j = 0; j <= l; ++j) {
      const double fj = diag[j];
      const double gj = subdiag[j];
      for (size_t k = j; k <= l; ++k) {
        zAt(k, j) -= (fj * subdiag[k]) + (gj * diag[k]);
      }
      diag[j] = zAt(l, j);
      zAt(ui, j) = 0.0;
    }
    diag[ui] = h;
  }

  for (size_t i = 1; i < n; ++i) {
    const size_t l = i - 1;
    zAt(n - 1, l) = zAt(l, l);
    zAt(l, l) = 1.0;
    h = diag[i];
    if (h != 0.0) {
      for (size_t k = 0; k <= l; ++k) {
        diag[k] = zAt(k, i) / h;
      }
      for (size_t j = 0; j <= l; ++j) {
        double g = 0.0;
        for (size_t k = 0; k <= l; ++k) {
          g += zAt(k, i) * zAt(k, j);
        }
        for (size_t k = 0; k <= l; ++k) {
          zAt(k, j) -= g * diag[k];
        }
      }
    }
    for (size_t k = 0; k <= l; ++k) {
      zAt(k, i) = 0.0;
    }
  }

  for (size_t j = 0; j < n; ++j) {
    diag[j] = zAt(n - 1, j);
  }
  for (size_t j = 0; j < n - 1; ++j) {
    zAt(n - 1, j) = 0.0;
  }
  zAt(n - 1, n - 1) = 1.0;
  subdiag[0] = 0.0;
}

/// EISPACK `tql2` for `n = 4` (column-major `z[row + col*n]`).
static void symmetricTql24(std::array<double, 4>& diag,
                           std::array<double, 4>& subdiag,
                           std::array<double, 16>& z) {
  constexpr size_t n = 4;
  const auto zAt = [&z](const size_t row, const size_t col) -> double& {
    return z[row + (col * n)];
  };

  for (size_t i = 1; i < n; ++i) {
    subdiag[i - 1] = subdiag[i];
  }
  double f = 0.0;
  double tst1 = 0.0;
  subdiag[n - 1] = 0.0;

  for (size_t l = 0; l < n; ++l) {
    int j = 0;
    const double h = std::abs(diag[l]) + std::abs(subdiag[l]);
    tst1 = std::max(tst1, h);

    size_t m = l;
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

        const size_t l1 = l + 1;
        const size_t l2 = l1 + 1;
        const double g = diag[l];
        const double p = (diag[l1] - g) / (2.0 * subdiag[l]);
        const double r = std::hypot(p, 1.0);
        diag[l] = subdiag[l] / (p + std::copysign(std::abs(r), p));
        diag[l1] = subdiag[l] * (p + std::copysign(std::abs(r), p));
        const double dl1 = diag[l1];
        const double hh = g - diag[l];
        for (size_t i = l2; i < n; ++i) {
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
        const size_t mml = m - l;
        for (size_t ii = 1; ii <= mml; ++ii) {
          c3 = c2;
          c2 = c;
          s2 = s;
          const size_t i = m - ii;
          const double gi = c * subdiag[i];
          const double hi = c * pv;
          const double ri = std::hypot(pv, subdiag[i]);
          subdiag[i + 1] = s * ri;
          s = subdiag[i] / ri;
          c = pv / ri;
          pv = (c * diag[i]) - (s * gi);
          diag[i + 1] = hi + (s * ((c * gi) + (s * diag[i])));
          for (size_t k = 0; k < n; ++k) {
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

  for (size_t ii = 1; ii < n; ++ii) {
    const size_t i = ii - 1;
    size_t k = i;
    double p = diag[i];
    for (size_t j = ii; j < n; ++j) {
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
    for (size_t j = 0; j < n; ++j) {
      const double tmp = zAt(j, i);
      zAt(j, i) = zAt(j, k);
      zAt(j, k) = tmp;
    }
  }
}

/**
 * @brief Computes the eigendecomposition of a real symmetric `4x4` matrix.
 *
 * Uses Householder tridiagonalization (EISPACK `tred2`) followed by implicit
 * QL iteration (`tql2`) on the tridiagonal form. Adapted from John Burkardt's
 * MIT-licensed EISPACK C port (`tred2`, `tql2`):
 * https://people.sc.fsu.edu/~jburkardt/c_src/eispack/eispack.c
 * Original Fortran: https://netlib.org/eispack/tred2.f,
 * https://netlib.org/eispack/tql2.f
 *
 * @pre @p symmetric has length `16` and forms a real symmetric matrix in
 * row-major order: `symmetric[(i * 4) + j] == symmetric[(j * 4) + i]` for all
 * `i, j`. Only the lower triangle (including the diagonal) is read, but
 * supplying a non-symmetric matrix yields undefined numerical results.
 *
 * @param symmetric Row-major real symmetric `4x4` matrix (`16` entries).
 * @return Ascending eigenvalues and matching eigenvectors (as columns).
 */
[[nodiscard]] static SymmetricEigenDecomposition4x4
symmetricEigenDecomposition4x4(const ArrayRef<double> symmetric) {
  if (symmetric.size() != 16) {
    llvm::reportFatalInternalError(
        "symmetricEigenDecomposition4x4 expects 16 row-major entries");
  }
  constexpr size_t n = 4;

  SymmetricEigenDecomposition4x4 result;
  std::array<double, 16> z{};
  std::array<double, 4> subdiag{};
  symmetricTred24(symmetric, z, result.eigenvalues, subdiag);
  symmetricTql24(result.eigenvalues, subdiag, z);

  for (size_t col = 0; col < n; ++col) {
    for (size_t row = 0; row < n; ++row) {
      result.eigenvectors(row, col) = z[row + (col * n)];
    }
  }
  return result;
}

[[nodiscard]] static bool isFiniteComplex(const Complex& value) {
  return std::isfinite(value.real()) && std::isfinite(value.imag());
}

static void normalizeInPlace(const llvm::MutableArrayRef<Complex> values) {
  double sumSq = 0.0;
  for (const Complex& value : values) {
    sumSq += std::norm(value);
  }
  const double norm = std::sqrt(sumSq);
  if (norm <= MATRIX_TOLERANCE) {
    return;
  }
  for (Complex& value : values) {
    value /= norm;
  }
}

// Complex EISPACK helpers:
// - `pythag` and `csroot`: John Burkardt's MIT-licensed C port
//   https://people.sc.fsu.edu/~jburkardt/c_src/eispack/eispack.c
// - `cdiv`, `corth`, `comqr2`: NETLIB EISPACK Fortran
//   https://netlib.org/eispack/cdiv.f
//   https://netlib.org/eispack/corth.f
//   https://netlib.org/eispack/comqr2.f
// Local names: eigenDecompositionStableHypot (pythag),
// eigenDecompositionComplexSqrt (csroot), eigenDecompositionComplexDivide
// (cdiv), eigenDecompositionReduceToHessenberg (corth),
// eigenDecompositionQrSolve (comqr2).

namespace {

/// Row-major `ld x n` matrix view for EISPACK storage (`values[row + col *
/// ld]`).
class EispackMatrixView {
public:
  EispackMatrixView(MutableArrayRef<double> values, const int ld)
      : values_(values), ld_(ld) {}

  [[nodiscard]] static size_t rowMajorIndex(const int row, const int col,
                                            const int ld) {
    return static_cast<size_t>(row) +
           (static_cast<size_t>(col) * static_cast<size_t>(ld));
  }

  [[nodiscard]] double& at(const int row, const int col) {
    return values_[rowMajorIndex(row, col, ld_)];
  }

  [[nodiscard]] const double& at(const int row, const int col) const {
    return values_[rowMajorIndex(row, col, ld_)];
  }

private:
  MutableArrayRef<double> values_;
  int ld_;
};

} // namespace

[[nodiscard]] static double eigenDecompositionStableHypot(const double a,
                                                          const double b) {
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

[[nodiscard]] static std::pair<double, double>
eigenDecompositionComplexSqrt(const double xr, const double xi) {
  const double inputReal = xr;
  const double inputImag = xi;
  const double s =
      std::sqrt(0.5 * (eigenDecompositionStableHypot(inputReal, inputImag) +
                       std::abs(inputReal)));

  double yr = 0.0;
  double yi = 0.0;
  if (0.0 <= inputReal) {
    yr = s;
  }

  double sSign = s;
  if (inputImag < 0.0) {
    sSign = -s;
  }

  if (inputReal <= 0.0) {
    yi = sSign;
  }

  if (inputReal < 0.0) {
    yr = 0.5 * (inputImag / yi);
  } else if (0.0 < inputReal) {
    yi = 0.5 * (inputImag / yr);
  }
  return {yr, yi};
}

[[nodiscard]] static std::pair<double, double> eigenDecompositionComplexDivide(
    const double dividendReal, const double dividendImag,
    const double divisorReal, const double divisorImag) {
  const double s = std::abs(divisorReal) + std::abs(divisorImag);
  const double dividendRealScaled = dividendReal / s;
  const double dividendImagScaled = dividendImag / s;
  const double divisorRealScaled = divisorReal / s;
  const double divisorImagScaled = divisorImag / s;
  const double denom = (divisorRealScaled * divisorRealScaled) +
                       (divisorImagScaled * divisorImagScaled);
  return {((dividendRealScaled * divisorRealScaled) +
           (dividendImagScaled * divisorImagScaled)) /
              denom,
          ((dividendImagScaled * divisorRealScaled) -
           (dividendRealScaled * divisorImagScaled)) /
              denom};
}

static void eigenDecompositionReduceToHessenberg(
    const int leadingDim, const int order, const int rowLow, const int rowHigh,
    MutableArrayRef<double> matrixRealBuf,
    MutableArrayRef<double> matrixImagBuf,
    MutableArrayRef<double> householderRealBuf,
    MutableArrayRef<double> householderImagBuf) {
  EispackMatrixView matrixReal(matrixRealBuf, leadingDim);
  EispackMatrixView matrixImag(matrixImagBuf, leadingDim);
  const auto householderRealAt =
      [&householderRealBuf](const int index) -> double& {
    return householderRealBuf[static_cast<size_t>(index)];
  };
  const auto householderImagAt =
      [&householderImagBuf](const int index) -> double& {
    return householderImagBuf[static_cast<size_t>(index)];
  };

  const int kp1 = rowLow + 1;
  const int la = rowHigh - 1;
  if (la < kp1) {
    return;
  }

  for (int m = kp1; m <= la; ++m) {
    double h = 0.0;
    householderRealAt(m) = 0.0;
    householderImagAt(m) = 0.0;
    double scale = 0.0;
    const int subCol = m - 1;
    for (int i = m; i <= rowHigh; ++i) {
      scale += std::abs(matrixReal.at(i, subCol)) +
               std::abs(matrixImag.at(i, subCol));
    }

    if (scale == 0.0) {
      continue;
    }

    const int mp = m + rowHigh;
    for (int ii = m; ii <= rowHigh; ++ii) {
      const int i = mp - ii;
      householderRealAt(i) = matrixReal.at(i, subCol) / scale;
      householderImagAt(i) = matrixImag.at(i, subCol) / scale;
      h += (householderRealAt(i) * householderRealAt(i)) +
           (householderImagAt(i) * householderImagAt(i));
    }

    double g = std::sqrt(h);
    const double f = eigenDecompositionStableHypot(householderRealAt(m),
                                                   householderImagAt(m));
    if (f == 0.0) {
      householderRealAt(m) = g;
      matrixReal.at(m, subCol) = scale;
    } else {
      h += f * g;
      g /= f;
      householderRealAt(m) = (1.0 + g) * householderRealAt(m);
      householderImagAt(m) = (1.0 + g) * householderImagAt(m);
    }

    for (int j = m; j < order; ++j) {
      double fr = 0.0;
      double fi = 0.0;
      for (int ii = m; ii <= rowHigh; ++ii) {
        const int i = mp - ii;
        fr += (householderRealAt(i) * matrixReal.at(i, j)) +
              (householderImagAt(i) * matrixImag.at(i, j));
        fi += (householderRealAt(i) * matrixImag.at(i, j)) -
              (householderImagAt(i) * matrixReal.at(i, j));
      }
      fr /= h;
      fi /= h;
      for (int i = m; i <= rowHigh; ++i) {
        matrixReal.at(i, j) -=
            (fr * householderRealAt(i)) - (fi * householderImagAt(i));
        matrixImag.at(i, j) -=
            (fr * householderImagAt(i)) + (fi * householderRealAt(i));
      }
    }

    for (int i = 0; i <= rowHigh; ++i) {
      double fr = 0.0;
      double fi = 0.0;
      for (int jj = m; jj <= rowHigh; ++jj) {
        const int j = mp - jj;
        fr += (householderRealAt(j) * matrixReal.at(i, j)) -
              (householderImagAt(j) * matrixImag.at(i, j));
        fi += (householderRealAt(j) * matrixImag.at(i, j)) +
              (householderImagAt(j) * matrixReal.at(i, j));
      }
      fr /= h;
      fi /= h;
      for (int j = m; j <= rowHigh; ++j) {
        matrixReal.at(i, j) -=
            (fr * householderRealAt(j)) + (fi * householderImagAt(j));
        matrixImag.at(i, j) +=
            (fr * householderImagAt(j)) - (fi * householderRealAt(j));
      }
    }

    householderRealAt(m) = scale * householderRealAt(m);
    householderImagAt(m) = scale * householderImagAt(m);
    matrixReal.at(m, subCol) = -g * matrixReal.at(m, subCol);
    matrixImag.at(m, subCol) = -g * matrixImag.at(m, subCol);
  }
}

[[nodiscard]] static int
eigenDecompositionQrSolve(const int leadingDim, const int order,
                          const int rowLow, const int rowHigh,
                          MutableArrayRef<double> householderRealBuf,
                          MutableArrayRef<double> householderImagBuf,
                          MutableArrayRef<double> hessenbergRealBuf,
                          MutableArrayRef<double> hessenbergImagBuf,
                          MutableArrayRef<double> eigenvalueRealBuf,
                          MutableArrayRef<double> eigenvalueImagBuf,
                          MutableArrayRef<double> eigenvectorRealBuf,
                          MutableArrayRef<double> eigenvectorImagBuf) {
  EispackMatrixView hessenbergReal(hessenbergRealBuf, leadingDim);
  EispackMatrixView hessenbergImag(hessenbergImagBuf, leadingDim);
  EispackMatrixView eigenvectorReal(eigenvectorRealBuf, leadingDim);
  EispackMatrixView eigenvectorImag(eigenvectorImagBuf, leadingDim);
  const auto householderRealAt =
      [&householderRealBuf](const int index) -> double& {
    return householderRealBuf[static_cast<size_t>(index)];
  };
  const auto householderImagAt =
      [&householderImagBuf](const int index) -> double& {
    return householderImagBuf[static_cast<size_t>(index)];
  };
  const auto eigenvalueRealAt =
      [&eigenvalueRealBuf](const int index) -> double& {
    return eigenvalueRealBuf[static_cast<size_t>(index)];
  };
  const auto eigenvalueImagAt =
      [&eigenvalueImagBuf](const int index) -> double& {
    return eigenvalueImagBuf[static_cast<size_t>(index)];
  };

  for (int j = 0; j < order; ++j) {
    for (int i = 0; i < order; ++i) {
      eigenvectorReal.at(i, j) = 0.0;
      eigenvectorImag.at(i, j) = 0.0;
    }
    eigenvectorReal.at(j, j) = 1.0;
  }

  const int numHouseholderReflections = rowHigh - rowLow - 1;
  if (numHouseholderReflections > 0) {
    for (int ii = 1; ii <= numHouseholderReflections; ++ii) {
      const int i = rowHigh - ii;
      if (householderRealAt(i) == 0.0 && householderImagAt(i) == 0.0) {
        continue;
      }
      if (hessenbergReal.at(i, i - 1) == 0.0 &&
          hessenbergImag.at(i, i - 1) == 0.0) {
        continue;
      }
      const double householderNorm =
          (hessenbergReal.at(i, i - 1) * householderRealAt(i)) +
          (hessenbergImag.at(i, i - 1) * householderImagAt(i));
      const int ip1 = i + 1;
      for (int k = ip1; k <= rowHigh; ++k) {
        householderRealAt(k) = hessenbergReal.at(k, i - 1);
        householderImagAt(k) = hessenbergImag.at(k, i - 1);
      }
      for (int j = i; j <= rowHigh; ++j) {
        double sr = 0.0;
        double si = 0.0;
        for (int k = i; k <= rowHigh; ++k) {
          sr += (householderRealAt(k) * eigenvectorReal.at(k, j)) +
                (householderImagAt(k) * eigenvectorImag.at(k, j));
          si += (householderRealAt(k) * eigenvectorImag.at(k, j)) -
                (householderImagAt(k) * eigenvectorReal.at(k, j));
        }
        sr /= householderNorm;
        si /= householderNorm;
        for (int k = i; k <= rowHigh; ++k) {
          eigenvectorReal.at(k, j) +=
              (sr * householderRealAt(k)) - (si * householderImagAt(k));
          eigenvectorImag.at(k, j) +=
              (sr * householderImagAt(k)) + (si * householderRealAt(k));
        }
      }
    }
  }

  if (numHouseholderReflections >= 0) {
    const int hessLow = rowLow + 1;
    for (int i = hessLow; i <= rowHigh; ++i) {
      const int ll = std::min(i + 1, rowHigh);
      if (hessenbergImag.at(i, i - 1) == 0.0) {
        continue;
      }
      const double subdiagonalNorm = eigenDecompositionStableHypot(
          hessenbergReal.at(i, i - 1), hessenbergImag.at(i, i - 1));
      const double yr = hessenbergReal.at(i, i - 1) / subdiagonalNorm;
      const double yi = hessenbergImag.at(i, i - 1) / subdiagonalNorm;
      hessenbergReal.at(i, i - 1) = subdiagonalNorm;
      hessenbergImag.at(i, i - 1) = 0.0;
      for (int j = i; j < order; ++j) {
        const double si =
            (yr * hessenbergImag.at(i, j)) - (yi * hessenbergReal.at(i, j));
        hessenbergReal.at(i, j) =
            (yr * hessenbergReal.at(i, j)) + (yi * hessenbergImag.at(i, j));
        hessenbergImag.at(i, j) = si;
      }
      for (int j = 0; j <= ll; ++j) {
        const double si =
            (yr * hessenbergImag.at(j, i)) + (yi * hessenbergReal.at(j, i));
        hessenbergReal.at(j, i) =
            (yr * hessenbergReal.at(j, i)) - (yi * hessenbergImag.at(j, i));
        hessenbergImag.at(j, i) = si;
      }
      for (int j = rowLow; j <= rowHigh; ++j) {
        const double si =
            (yr * eigenvectorImag.at(j, i)) + (yi * eigenvectorReal.at(j, i));
        eigenvectorReal.at(j, i) =
            (yr * eigenvectorReal.at(j, i)) - (yi * eigenvectorImag.at(j, i));
        eigenvectorImag.at(j, i) = si;
      }
    }
  }

  for (int i = 0; i < order; ++i) {
    if (i >= rowLow && i <= rowHigh) {
      continue;
    }
    eigenvalueRealAt(i) = hessenbergReal.at(i, i);
    eigenvalueImagAt(i) = hessenbergImag.at(i, i);
  }

  int activeEigenIndex = rowHigh;
  double eigenSumReal = 0.0;
  double eigenSumImag = 0.0;
  int qrIterationBudget = 30 * order;

  while (activeEigenIndex >= rowLow) {
    int qrIterationStep = 0;
    const int activeEigenIndexMinus1 = activeEigenIndex - 1;

    while (true) {
      int l = rowLow;
      for (int ll = rowLow; ll <= activeEigenIndex; ++ll) {
        l = activeEigenIndex + rowLow - ll;
        if (l == rowLow) {
          break;
        }
        const double tst1Local = std::abs(hessenbergReal.at((l - 1), l - 1)) +
                                 std::abs(hessenbergImag.at((l - 1), l - 1)) +
                                 std::abs(hessenbergReal.at(l, l)) +
                                 std::abs(hessenbergImag.at(l, l));
        const double tst2Local =
            tst1Local + std::abs(hessenbergReal.at(l, l - 1));
        if (tst2Local == tst1Local) {
          break;
        }
      }

      if (l == activeEigenIndex) {
        break;
      }
      if (qrIterationBudget == 0) {
        return activeEigenIndex;
      }

      double sr = hessenbergReal.at(activeEigenIndex, activeEigenIndex);
      double si = hessenbergImag.at(activeEigenIndex, activeEigenIndex);
      if (qrIterationStep == 10 || qrIterationStep == 20) {
        sr = std::abs(
            hessenbergReal.at(activeEigenIndex, activeEigenIndexMinus1));
        if (activeEigenIndex >= rowLow + 2) {
          sr += std::abs(
              hessenbergReal.at(activeEigenIndexMinus1, activeEigenIndex - 2));
        }
        si = 0.0;
      } else {
        double xr =
            hessenbergReal.at(activeEigenIndexMinus1, activeEigenIndex) *
            hessenbergReal.at(activeEigenIndex, activeEigenIndexMinus1);
        double xi =
            hessenbergImag.at(activeEigenIndexMinus1, activeEigenIndex) *
            hessenbergReal.at(activeEigenIndex, activeEigenIndexMinus1);
        if (xr != 0.0 || xi != 0.0) {
          const double yr = (hessenbergReal.at(activeEigenIndexMinus1,
                                               activeEigenIndexMinus1) -
                             sr) /
                            2.0;
          const double yi = (hessenbergImag.at(activeEigenIndexMinus1,
                                               activeEigenIndexMinus1) -
                             si) /
                            2.0;
          auto [zzr, zzi] = eigenDecompositionComplexSqrt(
              (yr * yr) - (yi * yi) + xr, (2.0 * yr * yi) + xi);
          if ((yr * zzr) + (yi * zzi) < 0.0) {
            zzr = -zzr;
            zzi = -zzi;
          }
          std::tie(xr, xi) =
              eigenDecompositionComplexDivide(xr, xi, yr + zzr, yi + zzi);
          sr -= xr;
          si -= xi;
        }
      }

      for (int i = rowLow; i <= activeEigenIndex; ++i) {
        hessenbergReal.at(i, i) -= sr;
        hessenbergImag.at(i, i) -= si;
      }
      eigenSumReal += sr;
      eigenSumImag += si;
      ++qrIterationStep;
      --qrIterationBudget;

      {
        const int lp1 = l + 1;
        for (int i = lp1; i <= activeEigenIndex; ++i) {
          sr = hessenbergReal.at(i, i - 1);
          hessenbergReal.at(i, i - 1) = 0.0;
          const double stepNorm = eigenDecompositionStableHypot(
              eigenDecompositionStableHypot(hessenbergReal.at((i - 1), i - 1),
                                            hessenbergImag.at((i - 1), i - 1)),
              sr);
          const double xr = hessenbergReal.at((i - 1), i - 1) / stepNorm;
          eigenvalueRealAt(i - 1) = xr;
          const double xi = hessenbergImag.at((i - 1), i - 1) / stepNorm;
          eigenvalueImagAt(i - 1) = xi;
          hessenbergReal.at((i - 1), i - 1) = stepNorm;
          hessenbergImag.at((i - 1), i - 1) = 0.0;
          hessenbergImag.at(i, i - 1) = sr / stepNorm;

          for (int j = i; j < order; ++j) {
            const double yr = hessenbergReal.at((i - 1), j);
            const double yi = hessenbergImag.at((i - 1), j);
            const double zzr = hessenbergReal.at(i, j);
            const double zzi = hessenbergImag.at(i, j);
            hessenbergReal.at((i - 1), j) =
                (xr * yr) + (xi * yi) + (hessenbergImag.at(i, i - 1) * zzr);
            hessenbergImag.at((i - 1), j) =
                (xr * yi) - (xi * yr) + (hessenbergImag.at(i, i - 1) * zzi);
            hessenbergReal.at(i, j) =
                (xr * zzr) - (xi * zzi) - (hessenbergImag.at(i, i - 1) * yr);
            hessenbergImag.at(i, j) =
                (xr * zzi) + (xi * zzr) - (hessenbergImag.at(i, i - 1) * yi);
          }
        }
      }

      si = hessenbergImag.at(activeEigenIndex, activeEigenIndex);
      if (si != 0.0) {
        const double stepNorm = eigenDecompositionStableHypot(
            hessenbergReal.at(activeEigenIndex, activeEigenIndex), si);
        sr = hessenbergReal.at(activeEigenIndex, activeEigenIndex) / stepNorm;
        si /= stepNorm;
        hessenbergReal.at(activeEigenIndex, activeEigenIndex) = stepNorm;
        hessenbergImag.at(activeEigenIndex, activeEigenIndex) = 0.0;
        if (activeEigenIndex != order - 1) {
          const int ip1 = activeEigenIndex + 1;
          for (int j = ip1; j < order; ++j) {
            const double yr = hessenbergReal.at(activeEigenIndex, j);
            const double yi = hessenbergImag.at(activeEigenIndex, j);
            hessenbergReal.at(activeEigenIndex, j) = (sr * yr) + (si * yi);
            hessenbergImag.at(activeEigenIndex, j) = (sr * yi) - (si * yr);
          }
        }
      }

      {
        const int lp1 = l + 1;
        for (int j = lp1; j <= activeEigenIndex; ++j) {
          const double xr = eigenvalueRealAt(j - 1);
          const double xi = eigenvalueImagAt(j - 1);
          for (int i = 0; i <= j; ++i) {
            const double yr = hessenbergReal.at(i, j - 1);
            double yi = 0.0;
            const double zzr = hessenbergReal.at(i, j);
            const double zzi = hessenbergImag.at(i, j);
            if (i != j) {
              yi = hessenbergImag.at(i, j - 1);
              hessenbergImag.at(i, j - 1) =
                  (xr * yi) + (xi * yr) + (hessenbergImag.at(j, j - 1) * zzi);
            }
            hessenbergReal.at(i, j - 1) =
                (xr * yr) - (xi * yi) + (hessenbergImag.at(j, j - 1) * zzr);
            hessenbergReal.at(i, j) =
                (xr * zzr) + (xi * zzi) - (hessenbergImag.at(j, j - 1) * yr);
            hessenbergImag.at(i, j) =
                (xr * zzi) - (xi * zzr) - (hessenbergImag.at(j, j - 1) * yi);
          }
          for (int i = rowLow; i <= rowHigh; ++i) {
            const double yr = eigenvectorReal.at(i, j - 1);
            const double yi = eigenvectorImag.at(i, j - 1);
            const double zzr = eigenvectorReal.at(i, j);
            const double zzi = eigenvectorImag.at(i, j);
            eigenvectorReal.at(i, j - 1) =
                (xr * yr) - (xi * yi) + (hessenbergImag.at(j, j - 1) * zzr);
            eigenvectorImag.at(i, j - 1) =
                (xr * yi) + (xi * yr) + (hessenbergImag.at(j, j - 1) * zzi);
            eigenvectorReal.at(i, j) =
                (xr * zzr) + (xi * zzi) - (hessenbergImag.at(j, j - 1) * yr);
            eigenvectorImag.at(i, j) =
                (xr * zzi) - (xi * zzr) - (hessenbergImag.at(j, j - 1) * yi);
          }
        }
      }

      if (si != 0.0) {
        for (int i = 0; i <= activeEigenIndex; ++i) {
          const double yr = hessenbergReal.at(i, activeEigenIndex);
          const double yi = hessenbergImag.at(i, activeEigenIndex);
          hessenbergReal.at(i, activeEigenIndex) = (sr * yr) - (si * yi);
          hessenbergImag.at(i, activeEigenIndex) = (sr * yi) + (si * yr);
        }
        for (int i = rowLow; i <= rowHigh; ++i) {
          const double yr = eigenvectorReal.at(i, activeEigenIndex);
          const double yi = eigenvectorImag.at(i, activeEigenIndex);
          eigenvectorReal.at(i, activeEigenIndex) = (sr * yr) - (si * yi);
          eigenvectorImag.at(i, activeEigenIndex) = (sr * yi) + (si * yr);
        }
      }
    }

    hessenbergReal.at(activeEigenIndex, activeEigenIndex) += eigenSumReal;
    eigenvalueRealAt(activeEigenIndex) =
        hessenbergReal.at(activeEigenIndex, activeEigenIndex);
    hessenbergImag.at(activeEigenIndex, activeEigenIndex) += eigenSumImag;
    eigenvalueImagAt(activeEigenIndex) =
        hessenbergImag.at(activeEigenIndex, activeEigenIndex);
    activeEigenIndex = activeEigenIndexMinus1;
  }

  double norm = 0.0;
  for (int i = 0; i < order; ++i) {
    for (int j = i; j < order; ++j) {
      const double matrixNorm =
          std::abs(hessenbergReal.at(i, j)) + std::abs(hessenbergImag.at(i, j));
      norm = std::max(norm, matrixNorm);
    }
  }

  if (order != 1 && norm != 0.0) {
    for (int nn = 2; nn <= order; ++nn) {
      activeEigenIndex = order + 1 - nn;
      double xr = eigenvalueRealAt(activeEigenIndex);
      double xi = eigenvalueImagAt(activeEigenIndex);
      hessenbergReal.at(activeEigenIndex, activeEigenIndex) = 1.0;
      hessenbergImag.at(activeEigenIndex, activeEigenIndex) = 0.0;
      for (int ii = 1; ii <= activeEigenIndex; ++ii) {
        const int i = activeEigenIndex - ii;
        double zzr = 0.0;
        double zzi = 0.0;
        const int ip1 = i + 1;
        for (int j = ip1; j <= activeEigenIndex; ++j) {
          zzr += (hessenbergReal.at(i, j) *
                  hessenbergReal.at(j, activeEigenIndex)) -
                 (hessenbergImag.at(i, j) *
                  hessenbergImag.at(j, activeEigenIndex));
          zzi += (hessenbergReal.at(i, j) *
                  hessenbergImag.at(j, activeEigenIndex)) +
                 (hessenbergImag.at(i, j) *
                  hessenbergReal.at(j, activeEigenIndex));
        }
        double yr = xr - eigenvalueRealAt(i);
        double yi = xi - eigenvalueImagAt(i);
        if (yr == 0.0 && yi == 0.0) {
          double tst1 = norm;
          yr = tst1;
          double tst2 = 0.0;
          do {
            yr = 0.01 * yr;
            tst2 = norm + yr;
          } while (tst2 <= tst1);
        }
        auto [divReal, divImag] =
            eigenDecompositionComplexDivide(zzr, zzi, yr, yi);
        hessenbergReal.at(i, activeEigenIndex) = divReal;
        hessenbergImag.at(i, activeEigenIndex) = divImag;
        const double trLocal =
            std::abs(hessenbergReal.at(i, activeEigenIndex)) +
            std::abs(hessenbergImag.at(i, activeEigenIndex));
        if (trLocal == 0.0) {
          continue;
        }
        const double tst1 = trLocal;
        const double tst2 = tst1 + (1.0 / tst1);
        if (tst2 > tst1) {
          continue;
        }
        for (int j = i; j <= activeEigenIndex; ++j) {
          hessenbergReal.at(j, activeEigenIndex) /= trLocal;
          hessenbergImag.at(j, activeEigenIndex) /= trLocal;
        }
      }
    }

    for (int i = 0; i < order; ++i) {
      if (i >= rowLow && i <= rowHigh) {
        continue;
      }
      for (int j = i; j < order; ++j) {
        eigenvectorReal.at(i, j) = hessenbergReal.at(i, j);
        eigenvectorImag.at(i, j) = hessenbergImag.at(i, j);
      }
    }

    for (int jj = rowLow; jj <= rowHigh; ++jj) {
      const int j = rowHigh + rowLow - jj;
      const int m = std::min(j, rowHigh);
      for (int i = rowLow; i <= rowHigh; ++i) {
        double zzr = 0.0;
        double zzi = 0.0;
        for (int k = rowLow; k <= m; ++k) {
          zzr += (eigenvectorReal.at(i, k) * hessenbergReal.at(k, j)) -
                 (eigenvectorImag.at(i, k) * hessenbergImag.at(k, j));
          zzi += (eigenvectorReal.at(i, k) * hessenbergImag.at(k, j)) +
                 (eigenvectorImag.at(i, k) * hessenbergReal.at(k, j));
        }
        eigenvectorReal.at(i, j) = zzr;
        eigenvectorImag.at(i, j) = zzi;
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

[[nodiscard]] static double
eigenvectorColumnNorm(const int order, const int col, const int leadingDim,
                      const ArrayRef<double> eigenvectorReal,
                      const ArrayRef<double> eigenvectorImag) {
  double normSq = 0.0;
  for (int row = 0; row < order; ++row) {
    const size_t idx = EispackMatrixView::rowMajorIndex(row, col, leadingDim);
    normSq += (eigenvectorReal[idx] * eigenvectorReal[idx]) +
              (eigenvectorImag[idx] * eigenvectorImag[idx]);
  }
  return std::sqrt(normSq);
}

[[nodiscard]] static Complex normalizedEigenvectorEntry(const double real,
                                                        const double imag,
                                                        const double norm) {
  if (norm > MATRIX_TOLERANCE) {
    return {real / norm, imag / norm};
  }
  return {real, imag};
}

[[nodiscard]] static EigenDecomposition4x4 assembleEigenDecomposition4x4(
    const std::array<double, K_COMPLEX_EIGEN4_EIGENVALUE_STORAGE>&
        eigenvalueReal,
    const std::array<double, K_COMPLEX_EIGEN4_EIGENVALUE_STORAGE>&
        eigenvalueImag,
    const std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE>& eigenvectorReal,
    const std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE>&
        eigenvectorImag) {
  EigenDecomposition4x4 result;
  for (int col = 0; col < K_COMPLEX_EIGEN4_SIZE; ++col) {
    result.eigenvalues[static_cast<size_t>(col)] =
        Complex(eigenvalueReal[static_cast<size_t>(col)],
                eigenvalueImag[static_cast<size_t>(col)]);
    const double norm =
        eigenvectorColumnNorm(K_COMPLEX_EIGEN4_SIZE, col, K_COMPLEX_EIGEN4_SIZE,
                              eigenvectorReal, eigenvectorImag);
    for (int row = 0; row < K_COMPLEX_EIGEN4_SIZE; ++row) {
      const size_t idx =
          EispackMatrixView::rowMajorIndex(row, col, K_COMPLEX_EIGEN4_SIZE);
      result.eigenvectors(static_cast<size_t>(row), static_cast<size_t>(col)) =
          normalizedEigenvectorEntry(eigenvectorReal[idx], eigenvectorImag[idx],
                                     norm);
    }
  }
  return result;
}

static void splitMatrix4x4ToRealImag(
    const Matrix4x4& matrix,
    std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE>& matrixReal,
    std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE>& matrixImag) {
  for (size_t row = 0; row < Matrix4x4::K_ROWS; ++row) {
    for (size_t col = 0; col < Matrix4x4::K_COLS; ++col) {
      const Complex& value = matrix(row, col);
      const size_t idx = row + (col * Matrix4x4::K_ROWS);
      matrixReal[idx] = std::real(value);
      matrixImag[idx] = std::imag(value);
    }
  }
}

/**
 * @brief Computes the eigendecomposition of a `4x4` complex matrix.
 *
 * Stack-specialized variant of the dynamic-matrix EISPACK solver for `n = 4`.
 *
 * @param matrix Source matrix.
 * @return Eigenpairs, or `std::nullopt` if the solver does not converge.
 */
[[nodiscard]] static std::optional<EigenDecomposition4x4>
eigenDecomposition4x4(const Matrix4x4& matrix) {
  constexpr int order = K_COMPLEX_EIGEN4_SIZE;
  constexpr int leadingDim = order;
  constexpr int rowLow = 0;
  constexpr int rowHigh = order - 1;

  std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE> matrixReal{};
  std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE> matrixImag{};
  splitMatrix4x4ToRealImag(matrix, matrixReal, matrixImag);

  std::array<double, 4> householderReal{};
  std::array<double, 4> householderImag{};
  std::array<double, K_COMPLEX_EIGEN4_EIGENVALUE_STORAGE> eigenvalueReal{};
  std::array<double, K_COMPLEX_EIGEN4_EIGENVALUE_STORAGE> eigenvalueImag{};
  std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE> eigenvectorReal{};
  std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE> eigenvectorImag{};

  eigenDecompositionReduceToHessenberg(leadingDim, order, rowLow, rowHigh,
                                       matrixReal, matrixImag, householderReal,
                                       householderImag);
  const int convergenceStatus = eigenDecompositionQrSolve(
      leadingDim, order, rowLow, rowHigh, householderReal, householderImag,
      matrixReal, matrixImag, eigenvalueReal, eigenvalueImag, eigenvectorReal,
      eigenvectorImag);
  if (convergenceStatus != 0) {
    return std::nullopt;
  }
  return assembleEigenDecomposition4x4(eigenvalueReal, eigenvalueImag,
                                       eigenvectorReal, eigenvectorImag);
}

/**
 * @brief Closed-form eigendecomposition of a `1x1` matrix.
 *
 * @param matrix Source matrix.
 * @return The single eigenpair.
 */
[[nodiscard]] static EigenDecomposition
eigenDecomposition1x1(const Matrix1x1& matrix) {
  EigenDecomposition result;
  result.eigenvalues.push_back(matrix.value);
  result.eigenvectors = DynamicMatrix(1);
  result.eigenvectors(0, 0) = 1.0;
  return result;
}

/**
 * @brief Computes the eigendecomposition of a `2x2` complex matrix using a
 * closed-form formula.
 *
 * @param matrix Source matrix.
 * @return Eigenpairs, or `std::nullopt` if the closed-form solver produces
 * non-finite eigenvalues.
 */
[[nodiscard]] static std::optional<EigenDecomposition2x2>
eigenDecomposition2x2(const Matrix2x2& matrix) {
  const Complex a = matrix(0, 0);
  const Complex b = matrix(0, 1);
  const Complex c = matrix(1, 0);
  const Complex d = matrix(1, 1);
  const Complex trace = a + d;
  const Complex determinant = a * d - b * c;
  const Complex discriminant = std::sqrt(trace * trace - 4.0 * determinant);
  const Complex lambda0 = (trace + discriminant) * 0.5;
  const Complex lambda1 = (trace - discriminant) * 0.5;
  if (!isFiniteComplex(lambda0) || !isFiniteComplex(lambda1)) {
    return std::nullopt;
  }

  if (std::abs(b) <= MATRIX_TOLERANCE && std::abs(c) <= MATRIX_TOLERANCE) {
    if (!isFiniteComplex(a) || !isFiniteComplex(d)) {
      return std::nullopt;
    }
    EigenDecomposition2x2 result;
    result.eigenvalues = {a, d};
    result.eigenvectors = Matrix2x2::identity();
    return result;
  }

  auto eigenvectorFor = [&](const Complex& lambda) -> SmallVector<Complex> {
    SmallVector<Complex> vector(2);
    if (std::abs(b) > MATRIX_TOLERANCE) {
      vector[0] = b;
      vector[1] = lambda - a;
    } else {
      vector[0] = lambda - d;
      vector[1] = c;
    }
    normalizeInPlace(vector);
    return vector;
  };

  const SmallVector<Complex> vector0 = eigenvectorFor(lambda0);
  const SmallVector<Complex> vector1 = eigenvectorFor(lambda1);

  EigenDecomposition2x2 result;
  result.eigenvalues = {lambda0, lambda1};
  result.eigenvectors(0, 0) = vector0[0];
  result.eigenvectors(1, 0) = vector0[1];
  result.eigenvectors(0, 1) = vector1[0];
  result.eigenvectors(1, 1) = vector1[1];
  return result;
}

/**
 * @brief EISPACK eigendecomposition for square dynamic matrices.
 *
 * For dimensions other than `1`, `2`, and `4`, which have specialized paths in
 * @ref DynamicMatrix::eigenDecomposition. Uses EISPACK `corth` followed by
 * `comqr2` (complex Hessenberg reduction and QR eigenanalysis). `pythag` and
 * `csroot` follow John Burkardt's MIT-licensed EISPACK C port; `cdiv`,
 * `corth`, and `comqr2` follow NETLIB EISPACK Fortran
 * (https://netlib.org/eispack/cdiv.f, https://netlib.org/eispack/corth.f,
 * https://netlib.org/eispack/comqr2.f). See also
 * https://people.sc.fsu.edu/~jburkardt/c_src/eispack/eispack.c
 *
 * @pre @p matrix has dimension at least `3` and not equal to `4`.
 *
 * @param matrix Square source matrix.
 * @return Eigenpairs, or `std::nullopt` if the matrix is not square, its
 * dimension exceeds `INT_MAX`, or the solver does not converge.
 */
[[nodiscard]] static std::optional<EigenDecomposition>
eigenDecompositionDynamic(const DynamicMatrix& matrix) {
  const int64_t dim = matrix.rows();
  if (dim != matrix.cols()) {
    return std::nullopt;
  }
  if (dim > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    return std::nullopt;
  }
  assert(dim >= 3 && dim != 4);
  const int order = static_cast<int>(dim);
  const int leadingDim = order;
  const int rowLow = 0;
  const int rowHigh = order - 1;

  const size_t matrixStorage =
      (static_cast<size_t>(leadingDim) * static_cast<size_t>(order)) +
      static_cast<size_t>(order) + 1U;
  SmallVector<double> matrixReal(matrixStorage);
  SmallVector<double> matrixImag(matrixStorage);
  for (int row = 0; row < order; ++row) {
    for (int col = 0; col < order; ++col) {
      const Complex value = matrix(row, col);
      const size_t idx = EispackMatrixView::rowMajorIndex(row, col, leadingDim);
      matrixReal[idx] = std::real(value);
      matrixImag[idx] = std::imag(value);
    }
  }

  SmallVector<double> householderReal(static_cast<size_t>(order));
  SmallVector<double> householderImag(static_cast<size_t>(order));
  SmallVector<double> eigenvalueReal(static_cast<size_t>(order) + 1U);
  SmallVector<double> eigenvalueImag(static_cast<size_t>(order) + 1U);
  SmallVector<double> eigenvectorReal(matrixStorage);
  SmallVector<double> eigenvectorImag(matrixStorage);

  eigenDecompositionReduceToHessenberg(leadingDim, order, rowLow, rowHigh,
                                       matrixReal, matrixImag, householderReal,
                                       householderImag);
  const int convergenceStatus = eigenDecompositionQrSolve(
      leadingDim, order, rowLow, rowHigh, householderReal, householderImag,
      matrixReal, matrixImag, eigenvalueReal, eigenvalueImag, eigenvectorReal,
      eigenvectorImag);
  if (convergenceStatus != 0) {
    return std::nullopt;
  }

  EigenDecomposition result;
  result.eigenvalues.reserve(static_cast<size_t>(order));
  result.eigenvectors = DynamicMatrix(dim);
  for (int col = 0; col < order; ++col) {
    result.eigenvalues.emplace_back(eigenvalueReal[static_cast<size_t>(col)],
                                    eigenvalueImag[static_cast<size_t>(col)]);
    const double norm = eigenvectorColumnNorm(order, col, leadingDim,
                                              eigenvectorReal, eigenvectorImag);
    for (int row = 0; row < order; ++row) {
      const size_t idx = EispackMatrixView::rowMajorIndex(row, col, leadingDim);
      result.eigenvectors(row, col) = normalizedEigenvectorEntry(
          eigenvectorReal[idx], eigenvectorImag[idx], norm);
    }
  }
  return result;
}

EigenDecomposition Matrix1x1::eigenDecomposition() const {
  return eigenDecomposition1x1(*this);
}

std::optional<EigenDecomposition2x2> Matrix2x2::eigenDecomposition() const {
  return eigenDecomposition2x2(*this);
}

Matrix4x4 Matrix4x4::fromRealRowMajor(const ArrayRef<double> entries) {
  if (entries.size() != K_SIZE_AT_COMPILE_TIME) {
    llvm::reportFatalInternalError(
        "Matrix4x4::fromRealRowMajor expects 16 row-major entries");
  }
  Matrix4x4 result;
  for (size_t i = 0; i < K_SIZE_AT_COMPILE_TIME; ++i) {
    result.data[i] = entries[i];
  }
  return result;
}

std::optional<EigenDecomposition4x4> Matrix4x4::eigenDecomposition() const {
  return eigenDecomposition4x4(*this);
}

SymmetricEigenDecomposition4x4 Matrix4x4::symmetricEigenDecomposition() const {
  return symmetricEigenDecomposition4x4(realPart());
}

struct DynamicMatrix::Impl {
  int64_t dim = 0;
  SmallVector<Complex> data;
};

DynamicMatrix::DynamicMatrix() : impl_(std::make_unique<Impl>()) {}

DynamicMatrix::DynamicMatrix(const int64_t dim)
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

int64_t DynamicMatrix::rows() const { return impl_->dim; }

int64_t DynamicMatrix::cols() const { return impl_->dim; }

DynamicMatrix DynamicMatrix::identity(const int64_t dim) {
  DynamicMatrix matrix(dim);
  const auto udim = checkedDim(dim);
  for (size_t i = 0; i < udim; ++i) {
    matrix.impl_->data[(i * udim) + i] = 1.0;
  }
  return matrix;
}

DynamicMatrix DynamicMatrix::fromAdjoint(const Matrix2x2& src) {
  return DynamicMatrix(src.adjoint());
}

Complex& DynamicMatrix::operator()(const int64_t row, const int64_t col) {
  return impl_
      ->data[static_cast<size_t>(checkedFlatIndex(row, col, impl_->dim))];
}

Complex DynamicMatrix::operator()(const int64_t row, const int64_t col) const {
  return impl_
      ->data[static_cast<size_t>(checkedFlatIndex(row, col, impl_->dim))];
}

void DynamicMatrix::setBottomRightCorner(const Matrix2x2& block) {
  copyBottomRightCorner(impl_->dim, impl_->data,
                        static_cast<int64_t>(Matrix2x2::K_ROWS), block.data);
}

void DynamicMatrix::setBottomRightCorner(const Matrix4x4& block) {
  copyBottomRightCorner(impl_->dim, impl_->data,
                        static_cast<int64_t>(Matrix4x4::K_ROWS), block.data);
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
  const ArrayRef<Complex> storage = impl_->data;
  for (size_t i = 0; i < udim; ++i) {
    sum += storage[(i * udim) + i];
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
  for (size_t row = 0; row < udim; ++row) {
    for (size_t col = 0; col < udim; ++col) {
      Complex sum{0.0, 0.0};
      for (size_t k = 0; k < udim; ++k) {
        sum +=
            impl_->data[(row * udim) + k] * rhs.impl_->data[(k * udim) + col];
      }
      out.impl_->data[(row * udim) + col] = sum;
    }
  }
  return out;
}

void DynamicMatrix::premultiplyBy(const DynamicMatrix& lhs) {
  *this = lhs * *this;
}

void DynamicMatrix::premultiplyByEmbedded1Q(const Matrix2x2& gate,
                                            const size_t numQubits,
                                            const size_t qubitIndex) {
  assert(qubitIndex < numQubits &&
         static_cast<uint64_t>(impl_->dim) == (uint64_t{1} << numQubits) &&
         "Matrix dimension must match numQubits");
  if (std::cmp_equal(impl_->dim, Matrix2x2::K_ROWS)) {
    assert(qubitIndex == 0);
    std::array<Complex, Matrix2x2::K_SIZE_AT_COMPILE_TIME> tmp{};
    multiply2x2(gate.data, impl_->data, tmp);
    impl_->data.assign(tmp.begin(), tmp.end());
    return;
  }
  const auto udim = checkedDim(impl_->dim);
  const size_t mask = size_t{1} << (numQubits - 1 - qubitIndex);
  const size_t step = mask << 1;
  auto& data = impl_->data;
  for (size_t chunk = 0; chunk < udim; chunk += step) {
    for (size_t inner = 0; inner < mask; ++inner) {
      const size_t base = chunk | inner;
      const size_t row1 = base | mask;
      for (size_t col = 0; col < udim; ++col) {
        const size_t idx0 = (base * udim) + col;
        const size_t idx1 = (row1 * udim) + col;
        apply2x2LeftToRowPair(gate.data, data[idx0], data[idx1]);
      }
    }
  }
}

void DynamicMatrix::premultiplyByEmbedded2Q(const Matrix4x4& gate,
                                            const size_t numQubits,
                                            const size_t q0Index,
                                            const size_t q1Index) {
  assert(q0Index < numQubits && q1Index < numQubits && q0Index != q1Index &&
         static_cast<uint64_t>(impl_->dim) == (uint64_t{1} << numQubits) &&
         "Matrix dimension must match numQubits");
  if (std::cmp_equal(impl_->dim, Matrix4x4::K_ROWS)) {
    assert(q0Index == 0 && q1Index == 1);
    std::array<Complex, Matrix4x4::K_SIZE_AT_COMPILE_TIME> tmp{};
    multiply4x4(gate.data, impl_->data, tmp);
    impl_->data.assign(tmp.begin(), tmp.end());
    return;
  }
  const auto udim = checkedDim(impl_->dim);
  const size_t mask0 = size_t{1} << (numQubits - 1 - q0Index);
  const size_t mask1 = size_t{1} << (numQubits - 1 - q1Index);
  auto& data = impl_->data;
  for (size_t block = 0; block < (udim >> 2); ++block) {
    size_t base = 0;
    size_t rest = block;
    for (size_t q = 0; q < numQubits; ++q) {
      if (q == q0Index || q == q1Index) {
        continue;
      }
      if ((rest & 1U) != 0) {
        base |= size_t{1} << (numQubits - 1 - q);
      }
      rest >>= 1U;
    }
    const std::array<size_t, Matrix4x4::K_ROWS> rowIdx = {
        base, base | mask1, base | mask0, base | mask0 | mask1};
    for (size_t col = 0; col < udim; ++col) {
      apply4x4LeftToColumn(gate.data, data[(rowIdx[0] * udim) + col],
                           data[(rowIdx[1] * udim) + col],
                           data[(rowIdx[2] * udim) + col],
                           data[(rowIdx[3] * udim) + col]);
    }
  }
}

DynamicMatrix DynamicMatrix::operator*(const Complex& scalar) const {
  DynamicMatrix out(impl_->dim);
  for (size_t i = 0; i < impl_->data.size(); ++i) {
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

EigenDecomposition
EigenDecomposition::from(const EigenDecomposition2x2& eigen2) {
  EigenDecomposition result;
  result.eigenvalues.assign(eigen2.eigenvalues.begin(),
                            eigen2.eigenvalues.end());
  result.eigenvectors = DynamicMatrix(eigen2.eigenvectors);
  return result;
}

EigenDecomposition
EigenDecomposition::from(const EigenDecomposition4x4& eigen4) {
  EigenDecomposition result;
  result.eigenvalues.assign(eigen4.eigenvalues.begin(),
                            eigen4.eigenvalues.end());
  result.eigenvectors = DynamicMatrix(eigen4.eigenvectors);
  return result;
}

std::optional<EigenDecomposition> DynamicMatrix::eigenDecomposition() const {
  const size_t dim = checkedDim(impl_->dim);
  if (dim == 0) {
    return std::nullopt;
  }
  if (dim == 1) {
    Matrix1x1 fixed;
    if (!fixed.assignFrom(*this)) {
      return std::nullopt;
    }
    return fixed.eigenDecomposition();
  }
  if (dim == 2) {
    Matrix2x2 fixed;
    if (!fixed.assignFrom(*this)) {
      return std::nullopt;
    }
    const std::optional<EigenDecomposition2x2> eigen2 =
        fixed.eigenDecomposition();
    if (!eigen2) {
      return std::nullopt;
    }
    return EigenDecomposition::from(*eigen2);
  }
  if (dim == 4) {
    Matrix4x4 fixed;
    if (!fixed.assignFrom(*this)) {
      return std::nullopt;
    }
    const std::optional<EigenDecomposition4x4> eigen4 =
        fixed.eigenDecomposition();
    if (!eigen4) {
      return std::nullopt;
    }
    return EigenDecomposition::from(*eigen4);
  }
  return eigenDecompositionDynamic(*this);
}

} // namespace mlir::qco
