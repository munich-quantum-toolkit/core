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

/**
 * @brief Returns the @p qubitIndex bit of a computational-basis label.
 *
 * Qubit @p i is bit @p i of @p stateIndex, matching @ref QuantumComputation.
 */
[[nodiscard]] static std::size_t qubitBitAt(const std::size_t stateIndex,
                                            const std::size_t qubitIndex) {
  return (stateIndex >> qubitIndex) & 1U;
}

/**
 * @brief True when row and col agree on every wire except @p skipA and @p
 * skipB.
 *
 * Untouched wires must match or the matrix entry is zero.
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
    if (qubitBitAt(row, q) != qubitBitAt(col, q)) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Packs the target-wire bits of @p stateIndex into a subspace index.
 *
 * Target wires are read in @p targetQubits order to index @p targetUnitary in
 * @ref embedControlledUnitary.
 */
[[nodiscard]] static std::size_t
extractTargetSubIndex(const std::size_t stateIndex,
                      const ArrayRef<std::size_t> targetQubits) {
  std::size_t sub = 0;
  for (const auto target : targetQubits) {
    sub = (sub << 1) | qubitBitAt(stateIndex, target);
  }
  return sub;
}

DynamicMatrix embedControlledUnitary(const std::size_t numQubits,
                                     const ArrayRef<std::size_t> controlQubits,
                                     const ArrayRef<std::size_t> targetQubits,
                                     const DynamicMatrix& targetUnitary) {
  assert(targetUnitary.rows() == targetUnitary.cols());
  assert(static_cast<std::size_t>(targetUnitary.rows()) ==
         (std::uint64_t{1} << targetQubits.size()));
  assert(numQubits < std::numeric_limits<std::size_t>::digits);
  for (const auto control : controlQubits) {
    assert(control < numQubits && "Control wire index out of range");
  }
  for (const auto target : targetQubits) {
    assert(target < numQubits && "Target wire index out of range");
  }

  const auto dim = checkedHilbertDim(numQubits);
  DynamicMatrix out = DynamicMatrix::identity(dim);
  const auto udim = static_cast<std::size_t>(dim);

  std::size_t activeMask = 0;
  for (const auto control : controlQubits) {
    activeMask |= std::size_t{1} << control;
  }
  const std::size_t controlMask = activeMask;
  for (const auto target : targetQubits) {
    activeMask |= std::size_t{1} << target;
  }
  // Wires outside the gate must match between row and col.
  const std::size_t passiveMask =
      ((std::size_t{1} << numQubits) - 1) & ~activeMask;

  llvm::SmallVector<std::int64_t, 64> targetIndexByState(udim);
  for (std::size_t state = 0; state < udim; ++state) {
    targetIndexByState[state] =
        static_cast<std::int64_t>(extractTargetSubIndex(state, targetQubits));
  }

  for (std::size_t row = 0; row < udim; ++row) {
    // Identity off the all-ones control subspace.
    if ((row & controlMask) != controlMask) {
      continue;
    }
    const std::int64_t targetRow = targetIndexByState[row];
    for (std::size_t col = 0; col < udim; ++col) {
      if ((col & controlMask) != controlMask ||
          ((row ^ col) & passiveMask) != 0) {
        continue;
      }
      out(static_cast<std::int64_t>(row), static_cast<std::int64_t>(col)) =
          targetUnitary(targetRow, targetIndexByState[col]);
    }
  }
  return out;
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
      const std::size_t rowBit = qubitBitAt(row, qubitIndex);
      const std::size_t colBit = qubitBitAt(col, qubitIndex);
      out(static_cast<std::int64_t>(row), static_cast<std::int64_t>(col)) =
          (*this)(rowBit, colBit);
    }
  }
  return out;
}

Matrix4x4 Matrix2x2::embedInTwoQubit(const std::size_t qubitIndex) const {
  if (qubitIndex == 0) {
    return Matrix4x4::kron(Matrix2x2::identity(), *this);
  }
  if (qubitIndex == 1) {
    return Matrix4x4::kron(*this, Matrix2x2::identity());
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
      const std::size_t rowPair =
          (qubitBitAt(row, q0Index) << 1) | qubitBitAt(row, q1Index);
      const std::size_t colPair =
          (qubitBitAt(col, q0Index) << 1) | qubitBitAt(col, q1Index);
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

} // namespace mlir::qco
