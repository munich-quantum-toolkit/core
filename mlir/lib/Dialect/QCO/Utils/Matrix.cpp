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

/// Returns true if every entry pair differs by at most @p tol (complex
/// modulus).
[[nodiscard]] static bool entriesAreApprox(ArrayRef<Complex> lhs,
                                           ArrayRef<Complex> rhs, double tol) {
  return std::ranges::equal(lhs, rhs,
                            [tol](const Complex& a, const Complex& b) {
                              return std::abs(a - b) <= tol;
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
static void
multiply2x2(const std::array<Complex, Matrix2x2::K_SIZE_AT_COMPILE_TIME>& lhs,
            const std::array<Complex, Matrix2x2::K_SIZE_AT_COMPILE_TIME>& rhs,
            std::array<Complex, Matrix2x2::K_SIZE_AT_COMPILE_TIME>& out) {
  out[0] = lhs[0] * rhs[0] + lhs[1] * rhs[2];
  out[1] = lhs[0] * rhs[1] + lhs[1] * rhs[3];
  out[2] = lhs[2] * rhs[0] + lhs[3] * rhs[2];
  out[3] = lhs[2] * rhs[1] + lhs[3] * rhs[3];
}

/// Writes the row-major product `lhs * rhs` into @p out (4x4, unrolled rows).
static void
multiply4x4(const std::array<Complex, Matrix4x4::K_SIZE_AT_COMPILE_TIME>& lhs,
            const std::array<Complex, Matrix4x4::K_SIZE_AT_COMPILE_TIME>& rhs,
            std::array<Complex, Matrix4x4::K_SIZE_AT_COMPILE_TIME>& out) {
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

struct DynamicMatrix::Impl {
  std::int64_t dim = 0;
  SmallVector<Complex> data;
};

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
  return std::abs(value - other.value) <= tol;
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

void Matrix2x2::premultiplyBy(const Matrix2x2& lhs) {
  const std::array<Complex, K_SIZE_AT_COMPILE_TIME> rhs = data;
  multiply2x2(lhs.data, rhs, data);
}

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

bool Matrix2x2::isApprox(const Matrix2x2& other, const double tol) const {
  return entriesAreApprox(data, other.data, tol);
}

bool Matrix2x2::isIdentity(const double tol) const {
  return isApprox(fromElements(1.0, 0.0, 0.0, 1.0), tol);
}

bool Matrix2x2::assignFrom(const DynamicMatrix& src) {
  return assignFromDynamicImpl<K_ROWS, K_SIZE_AT_COMPILE_TIME>(src, data);
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

void Matrix4x4::premultiplyBy(const Matrix4x4& lhs) {
  const std::array<Complex, K_SIZE_AT_COMPILE_TIME> rhs = data;
  multiply4x4(lhs.data, rhs, data);
}

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
  Matrix4x4 out{};
  for (std::size_t row = 0; row < K_ROWS; ++row) {
    for (std::size_t col = 0; col < K_COLS; ++col) {
      out.data[(col * K_COLS) + row] = data[(row * K_COLS) + col];
    }
  }
  return out;
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

bool Matrix4x4::isApprox(const Matrix4x4& other, const double tol) const {
  return entriesAreApprox(data, other.data, tol);
}

bool Matrix4x4::isIdentity(const double tol) const {
  Matrix4x4 id{};
  for (std::size_t i = 0; i < K_ROWS; ++i) {
    id.data[(i * K_COLS) + i] = 1.0;
  }
  return isApprox(id, tol);
}

std::array<Complex, Matrix4x4::K_ROWS> Matrix4x4::diagonal() const {
  return {data[0], data[5], data[10], data[15]};
}

Matrix4x4
Matrix4x4::fromDiagonal(const std::array<Complex, K_ROWS>& diagonalEntries) {
  Matrix4x4 out{};
  for (std::size_t i = 0; i < K_ROWS; ++i) {
    out.data[(i * K_COLS) + i] = diagonalEntries[i];
  }
  return out;
}

std::array<Complex, Matrix4x4::K_ROWS>
Matrix4x4::column(const std::size_t col) const {
  return {data[col], data[K_COLS + col], data[(2 * K_COLS) + col],
          data[(3 * K_COLS) + col]};
}

void Matrix4x4::setColumn(const std::size_t col,
                          const std::array<Complex, K_ROWS>& values) {
  for (std::size_t row = 0; row < K_ROWS; ++row) {
    data[(row * K_COLS) + col] = values[row];
  }
}

std::array<double, Matrix4x4::K_SIZE_AT_COMPILE_TIME>
Matrix4x4::realPart() const {
  std::array<double, K_SIZE_AT_COMPILE_TIME> out{};
  for (std::size_t i = 0; i < K_SIZE_AT_COMPILE_TIME; ++i) {
    out[i] = data[i].real();
  }
  return out;
}

std::array<double, Matrix4x4::K_SIZE_AT_COMPILE_TIME>
Matrix4x4::imagPart() const {
  std::array<double, K_SIZE_AT_COMPILE_TIME> out{};
  for (std::size_t i = 0; i < K_SIZE_AT_COMPILE_TIME; ++i) {
    out[i] = data[i].imag();
  }
  return out;
}

bool Matrix4x4::assignFrom(const DynamicMatrix& src) {
  return assignFromDynamicImpl<K_ROWS, K_SIZE_AT_COMPILE_TIME>(src, data);
}

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
  return std::abs(impl_->data[0] - other.value) <= tol;
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

Matrix2x2 operator*(const Complex& scalar, const Matrix2x2& matrix) {
  return matrix * scalar;
}

Matrix4x4 operator*(const Complex& scalar, const Matrix4x4& matrix) {
  return matrix * scalar;
}

Matrix4x4 kron(const Matrix2x2& lhs, const Matrix2x2& rhs) {
  Matrix4x4 out{};
  for (std::size_t i = 0; i < Matrix2x2::K_ROWS; ++i) {
    for (std::size_t j = 0; j < Matrix2x2::K_COLS; ++j) {
      const Complex a = lhs(i, j);
      for (std::size_t k = 0; k < Matrix2x2::K_ROWS; ++k) {
        for (std::size_t l = 0; l < Matrix2x2::K_COLS; ++l) {
          out((2 * i) + k, (2 * j) + l) = a * rhs(k, l);
        }
      }
    }
  }
  return out;
}

SymmetricEigen4 jacobiSymmetricEigen(const std::array<double, 16>& symmetric) {
  constexpr std::size_t n = 4;
  constexpr int maxSweeps = 100;

  std::array<double, 16> a = symmetric;
  std::array<double, 16> v{};
  for (std::size_t i = 0; i < n; ++i) {
    v[(i * n) + i] = 1.0;
  }

  for (int sweep = 0; sweep < maxSweeps; ++sweep) {
    double off = 0.0;
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        off += a[(p * n) + q] * a[(p * n) + q];
      }
    }
    if (off <= 1e-30) {
      break;
    }

    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        const double apq = a[(p * n) + q];
        if (std::abs(apq) <= 1e-300) {
          continue;
        }
        const double app = a[(p * n) + p];
        const double aqq = a[(q * n) + q];
        // Rotation angle that annihilates the (p, q) off-diagonal entry.
        const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(phi);
        const double s = std::sin(phi);

        // Right-multiply by the Givens rotation: columns p and q.
        for (std::size_t k = 0; k < n; ++k) {
          const double akp = a[(k * n) + p];
          const double akq = a[(k * n) + q];
          a[(k * n) + p] = (c * akp) - (s * akq);
          a[(k * n) + q] = (s * akp) + (c * akq);
        }
        // Left-multiply by the transposed rotation: rows p and q.
        for (std::size_t k = 0; k < n; ++k) {
          const double apk = a[(p * n) + k];
          const double aqk = a[(q * n) + k];
          a[(p * n) + k] = (c * apk) - (s * aqk);
          a[(q * n) + k] = (s * apk) + (c * aqk);
        }
        // Accumulate the rotation into the eigenvector matrix.
        for (std::size_t k = 0; k < n; ++k) {
          const double vkp = v[(k * n) + p];
          const double vkq = v[(k * n) + q];
          v[(k * n) + p] = (c * vkp) - (s * vkq);
          v[(k * n) + q] = (s * vkp) + (c * vkq);
        }
      }
    }
  }

  std::array<double, 4> evals{a[0], a[5], a[10], a[15]};
  std::array<std::size_t, 4> order{0, 1, 2, 3};
  std::ranges::sort(order, [&evals](const std::size_t x, const std::size_t y) {
    return evals[x] < evals[y];
  });

  SymmetricEigen4 result;
  for (std::size_t j = 0; j < n; ++j) {
    const std::size_t src = order[j];
    result.eigenvalues[j] = evals[src];
    for (std::size_t i = 0; i < n; ++i) {
      result.eigenvectors(i, j) = Complex{v[(i * n) + src], 0.0};
    }
  }
  return result;
}

} // namespace mlir::qco
