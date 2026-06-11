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
isApproxFixedImpl(const std::int64_t dim, ArrayRef<Complex> data,
                  const std::array<Complex, Size>& other, const double tol) {
  if (std::cmp_not_equal(dim, Dim)) {
    return false;
  }
  return entriesAreApprox(data, other, tol);
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
  return fromElements(data[0] * rhs.data[0] + data[1] * rhs.data[2],
                      data[0] * rhs.data[1] + data[1] * rhs.data[3],
                      data[2] * rhs.data[0] + data[3] * rhs.data[2],
                      data[2] * rhs.data[1] + data[3] * rhs.data[3]);
}

Matrix2x2 Matrix2x2::adjoint() const {
  return fromElements(std::conj(data[0]), std::conj(data[2]),
                      std::conj(data[1]), std::conj(data[3]));
}

Complex Matrix2x2::trace() const { return data[0] + data[3]; }

Complex Matrix2x2::determinant() const {
  return data[0] * data[3] - data[1] * data[2];
}

bool Matrix2x2::isApprox(const Matrix2x2& other, const double tol) const {
  return entriesAreApprox(data, other.data, tol);
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
  for (std::size_t row = 0; row < K_ROWS; ++row) {
    const std::size_t rowBase = row * K_COLS;
    const Complex& a0 = data[rowBase + 0];
    const Complex& a1 = data[rowBase + 1];
    const Complex& a2 = data[rowBase + 2];
    const Complex& a3 = data[rowBase + 3];
    out.data[rowBase + 0] = a0 * rhs.data[0] + a1 * rhs.data[4] +
                            a2 * rhs.data[8] + a3 * rhs.data[12];
    out.data[rowBase + 1] = a0 * rhs.data[1] + a1 * rhs.data[5] +
                            a2 * rhs.data[9] + a3 * rhs.data[13];
    out.data[rowBase + 2] = a0 * rhs.data[2] + a1 * rhs.data[6] +
                            a2 * rhs.data[10] + a3 * rhs.data[14];
    out.data[rowBase + 3] = a0 * rhs.data[3] + a1 * rhs.data[7] +
                            a2 * rhs.data[11] + a3 * rhs.data[15];
  }
  return out;
}

Matrix4x4 Matrix4x4::adjoint() const {
  Matrix4x4 out{};
  adjointInto(data, out.data, K_ROWS);
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

DynamicMatrix::DynamicMatrix() : impl_(std::make_unique<Impl>()) {}

DynamicMatrix::DynamicMatrix(const std::int64_t dim)
    : impl_(std::make_unique<Impl>()) {
  impl_->dim = dim;
  impl_->data.assign(checkedStorageSize(dim), Complex{});
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

} // namespace mlir::qco
