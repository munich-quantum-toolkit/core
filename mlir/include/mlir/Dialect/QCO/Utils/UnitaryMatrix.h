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

#include <llvm/ADT/SmallVector.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace mlir::qco {

/// Complex scalar type used for unitary matrix entries.
using Complex = std::complex<double>;

/// Default absolute tolerance for unitary matrix comparisons.
inline constexpr double UNITARY_MATRIX_TOLERANCE = 1e-14;

/**
 * @brief 1x1 unitary matrix for global-phase gates.
 *
 * Stores a single complex scalar in row-major layout. Used by operations such
 * as `GPhaseOp` whose unitary is a global phase factor.
 */
struct Matrix1x1 {
  /// Flat storage of the single matrix entry.
  std::array<Complex, 1> data{Complex{1.0, 0.0}};

  /**
   * @brief Constructs a matrix from its single entry.
   * @param m00 Element at row 0, column 0.
   * @return A new `Matrix1x1` with the given element.
   */
  [[nodiscard]] static Matrix1x1 fromElements(Complex m00) {
    Matrix1x1 m{};
    m(0, 0) = m00;
    return m;
  }

  /**
   * @brief Mutable element access with `(row, col)` indexing.
   * @param row Row index (ignored; always 0).
   * @param col Column index (ignored; always 0).
   * @return Reference to the sole matrix entry.
   */
  [[nodiscard]] Complex& operator()(std::size_t row, std::size_t col) {
    (void)row;
    (void)col;
    return data[0];
  }

  /**
   * @brief Const element access with `(row, col)` indexing.
   * @param row Row index (ignored; always 0).
   * @param col Column index (ignored; always 0).
   * @return Copy of the sole matrix entry.
   */
  [[nodiscard]] Complex operator()(std::size_t row, std::size_t col) const {
    (void)row;
    (void)col;
    return data[0];
  }

  /**
   * @brief Checks approximate equality using an absolute entry-wise tolerance.
   * @param other Matrix to compare against.
   * @param tol Maximum allowed absolute difference per entry.
   * @return True if the absolute difference is within @p tol.
   */
  [[nodiscard]] bool isApprox(const Matrix1x1& other,
                              double tol = UNITARY_MATRIX_TOLERANCE) const {
    return std::abs(data[0] - other.data[0]) <= tol;
  }
};

/**
 * @brief Fixed-size 2x2 unitary matrix in row-major layout.
 *
 * Represents single-qubit gate unitaries. Elements are stored in a flat array
 * with index `(row * K_COLS) + col`.
 */
struct Matrix2 {
  /// Number of rows.
  static constexpr std::size_t K_ROWS = 2;
  /// Number of columns.
  static constexpr std::size_t K_COLS = 2;
  /// Total number of stored elements.
  static constexpr std::size_t K_SIZE_AT_COMPILE_TIME = 4;

  /// Flat row-major storage of all matrix entries.
  std::array<Complex, 4> data{};

  /**
   * @brief Constructs a matrix from its four row-major entries.
   * @param m00 Element at row 0, column 0.
   * @param m01 Element at row 0, column 1.
   * @param m10 Element at row 1, column 0.
   * @param m11 Element at row 1, column 1.
   * @return A new `Matrix2` with the given elements.
   */
  [[nodiscard]] static Matrix2 fromElements(Complex m00, Complex m01,
                                            Complex m10, Complex m11) {
    Matrix2 m{};
    m.data = {m00, m01, m10, m11};
    return m;
  }

  /**
   * @brief Returns the 2x2 identity matrix.
   * @return Identity matrix `[[1, 0], [0, 1]]`.
   */
  [[nodiscard]] static Matrix2 identity() { return fromElements(1, 0, 0, 1); }

  /**
   * @brief Mutable element access.
   * @param row Row index in `[0, K_ROWS)`.
   * @param col Column index in `[0, K_COLS)`.
   * @return Reference to the element at `(row, col)`.
   */
  [[nodiscard]] Complex& operator()(std::size_t row, std::size_t col) {
    return data[(row * K_COLS) + col];
  }

  /**
   * @brief Const element access.
   * @param row Row index in `[0, K_ROWS)`.
   * @param col Column index in `[0, K_COLS)`.
   * @return Copy of the element at `(row, col)`.
   */
  [[nodiscard]] Complex operator()(std::size_t row, std::size_t col) const {
    return data[(row * K_COLS) + col];
  }

  /**
   * @brief Matrix product `*this * rhs`.
   * @param rhs Right-hand factor.
   * @return Product of the two matrices.
   */
  [[nodiscard]] Matrix2 operator*(const Matrix2& rhs) const {
    Matrix2 out{};
    for (std::size_t i = 0; i < K_ROWS; ++i) {
      for (std::size_t j = 0; j < K_COLS; ++j) {
        out(i, j) = (*this)(i, 0) * rhs(0, j) + (*this)(i, 1) * rhs(1, j);
      }
    }
    return out;
  }

  /**
   * @brief Returns the conjugate transpose (adjoint) of this matrix.
   * @return Adjoint matrix `A^\dagger`.
   */
  [[nodiscard]] Matrix2 adjoint() const {
    return fromElements(std::conj((*this)(0, 0)), std::conj((*this)(1, 0)),
                        std::conj((*this)(0, 1)), std::conj((*this)(1, 1)));
  }

  /**
   * @brief Returns the trace of this matrix.
   * @return Sum of diagonal entries.
   */
  [[nodiscard]] Complex trace() const { return (*this)(0, 0) + (*this)(1, 1); }

  /**
   * @brief Returns the determinant of this matrix.
   * @return Complex determinant `ad - bc`.
   */
  [[nodiscard]] Complex determinant() const {
    return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
  }

  /**
   * @brief Checks approximate equality using an absolute entry-wise tolerance.
   *
   * For each entry `i`, the comparison uses `|a_i - b_i| <= tol`, where the
   * absolute value is the complex modulus.
   *
   * @param other Matrix to compare against.
   * @param tol Maximum allowed absolute difference per entry.
   * @return True if every entry differs by at most @p tol.
   */
  [[nodiscard]] bool isApprox(const Matrix2& other,
                              double tol = UNITARY_MATRIX_TOLERANCE) const {
    for (std::size_t i = 0; i < data.size(); ++i) {
      if (std::abs(data[i] - other.data[i]) > tol) {
        return false;
      }
    }
    return true;
  }
};

/**
 * @brief Fixed-size 4x4 unitary matrix in row-major layout.
 *
 * Represents two-qubit gate unitaries. Elements are stored in a flat array with
 * index `(row * K_COLS) + col`.
 */
struct Matrix4 {
  /// Number of rows.
  static constexpr std::size_t K_ROWS = 4;
  /// Number of columns.
  static constexpr std::size_t K_COLS = 4;
  /// Total number of stored elements.
  static constexpr std::size_t K_SIZE_AT_COMPILE_TIME = 16;

  /// Flat row-major storage of all matrix entries.
  std::array<Complex, 16> data{};

  /**
   * @brief Constructs a matrix from its sixteen row-major entries.
   * @param m00 Element at row 0, column 0.
   * @param m01 Element at row 0, column 1.
   * @param m02 Element at row 0, column 2.
   * @param m03 Element at row 0, column 3.
   * @param m10 Element at row 1, column 0.
   * @param m11 Element at row 1, column 1.
   * @param m12 Element at row 1, column 2.
   * @param m13 Element at row 1, column 3.
   * @param m20 Element at row 2, column 0.
   * @param m21 Element at row 2, column 1.
   * @param m22 Element at row 2, column 2.
   * @param m23 Element at row 2, column 3.
   * @param m30 Element at row 3, column 0.
   * @param m31 Element at row 3, column 1.
   * @param m32 Element at row 3, column 2.
   * @param m33 Element at row 3, column 3.
   * @return A new `Matrix4` with the given elements.
   */
  [[nodiscard]] static Matrix4
  fromElements(Complex m00, Complex m01, Complex m02, Complex m03, Complex m10,
               Complex m11, Complex m12, Complex m13, Complex m20, Complex m21,
               Complex m22, Complex m23, Complex m30, Complex m31, Complex m32,
               Complex m33) {
    Matrix4 m{};
    m.data = {m00, m01, m02, m03, m10, m11, m12, m13,
              m20, m21, m22, m23, m30, m31, m32, m33};
    return m;
  }

  /**
   * @brief Returns the 4x4 identity matrix.
   * @return Identity matrix with ones on the diagonal.
   */
  [[nodiscard]] static Matrix4 identity() {
    Matrix4 m{};
    for (std::size_t i = 0; i < K_ROWS; ++i) {
      m(i, i) = 1.0;
    }
    return m;
  }

  /**
   * @brief Mutable element access.
   * @param row Row index in `[0, K_ROWS)`.
   * @param col Column index in `[0, K_COLS)`.
   * @return Reference to the element at `(row, col)`.
   */
  [[nodiscard]] Complex& operator()(std::size_t row, std::size_t col) {
    return data[(row * K_COLS) + col];
  }

  /**
   * @brief Const element access.
   * @param row Row index in `[0, K_ROWS)`.
   * @param col Column index in `[0, K_COLS)`.
   * @return Copy of the element at `(row, col)`.
   */
  [[nodiscard]] Complex operator()(std::size_t row, std::size_t col) const {
    return data[(row * K_COLS) + col];
  }

  /**
   * @brief Matrix product `*this * rhs`.
   * @param rhs Right-hand factor.
   * @return Product of the two matrices.
   */
  [[nodiscard]] Matrix4 operator*(const Matrix4& rhs) const {
    Matrix4 out{};
    for (std::size_t i = 0; i < K_ROWS; ++i) {
      for (std::size_t j = 0; j < K_COLS; ++j) {
        Complex sum{0.0, 0.0};
        for (std::size_t k = 0; k < K_COLS; ++k) {
          sum += (*this)(i, k) * rhs(k, j);
        }
        out(i, j) = sum;
      }
    }
    return out;
  }

  /**
   * @brief Returns the conjugate transpose (adjoint) of this matrix.
   * @return Adjoint matrix `A^\dagger`.
   */
  [[nodiscard]] Matrix4 adjoint() const {
    Matrix4 out{};
    for (std::size_t i = 0; i < K_ROWS; ++i) {
      for (std::size_t j = 0; j < K_COLS; ++j) {
        out(i, j) = std::conj((*this)(j, i));
      }
    }
    return out;
  }

  /**
   * @brief Returns the trace of this matrix.
   * @return Sum of diagonal entries.
   */
  [[nodiscard]] Complex trace() const {
    Complex t{0.0, 0.0};
    for (std::size_t i = 0; i < K_ROWS; ++i) {
      t += (*this)(i, i);
    }
    return t;
  }

  /**
   * @brief Returns the determinant of this matrix.
   * @return Complex determinant computed via Laplace expansion.
   */
  [[nodiscard]] Complex determinant() const;

  /**
   * @brief Checks approximate equality using an absolute entry-wise tolerance.
   * @param other Matrix to compare against.
   * @param tol Maximum allowed absolute difference per entry.
   * @return True if every entry differs by at most @p tol.
   */
  [[nodiscard]] bool isApprox(const Matrix4& other,
                              double tol = UNITARY_MATRIX_TOLERANCE) const {
    for (std::size_t i = 0; i < data.size(); ++i) {
      if (std::abs(data[i] - other.data[i]) > tol) {
        return false;
      }
    }
    return true;
  }
};

/**
 * @brief Square unitary matrix with runtime dimension.
 *
 * Used when the Hilbert-space dimension depends on the operation, for example
 * in controlled gates (`CtrlOp`) and inverses (`InvOp`). Storage is row-major
 * in a `llvm::SmallVector`.
 */
class DynamicMatrix {
public:
  /// Creates an empty 0x0 matrix.
  DynamicMatrix() = default;

  /**
   * @brief Creates a zero-initialized square matrix.
   * @param dim Side length of the square matrix.
   */
  explicit DynamicMatrix(std::int64_t dim) : dim_(dim), data_(dim * dim) {}

  /**
   * @brief Returns a square identity matrix of the given dimension.
   * @param dim Side length of the identity matrix.
   * @return Identity matrix with ones on the diagonal.
   */
  [[nodiscard]] static DynamicMatrix identity(std::int64_t dim) {
    DynamicMatrix m(dim);
    for (std::int64_t i = 0; i < dim; ++i) {
      m(i, i) = 1.0;
    }
    return m;
  }

  /**
   * @brief Returns the number of rows.
   * @return Matrix dimension.
   */
  [[nodiscard]] std::int64_t rows() const { return dim_; }

  /**
   * @brief Returns the number of columns.
   * @return Matrix dimension.
   */
  [[nodiscard]] std::int64_t cols() const { return dim_; }

  /**
   * @brief Mutable element access.
   * @param row Row index in `[0, dim)`.
   * @param col Column index in `[0, dim)`.
   * @return Reference to the element at `(row, col)`.
   */
  [[nodiscard]] Complex& operator()(std::int64_t row, std::int64_t col) {
    return data_[static_cast<std::size_t>((row * dim_) + col)];
  }

  /**
   * @brief Const element access.
   * @param row Row index in `[0, dim)`.
   * @param col Column index in `[0, dim)`.
   * @return Copy of the element at `(row, col)`.
   */
  [[nodiscard]] Complex operator()(std::int64_t row, std::int64_t col) const {
    return data_[static_cast<std::size_t>((row * dim_) + col)];
  }

  /**
   * @brief Copies a 2x2 block into the bottom-right corner.
   * @param block Source block placed at indices `(dim-2, dim-2)` through
   * `(dim-1, dim-1)`.
   */
  void setBottomRightCorner(const Matrix2& block) {
    const std::int64_t offset = dim_ - 2;
    for (std::int64_t i = 0; i < 2; ++i) {
      for (std::int64_t j = 0; j < 2; ++j) {
        (*this)(offset + i, offset + j) =
            block(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
      }
    }
  }

  /**
   * @brief Copies a 4x4 block into the bottom-right corner.
   * @param block Source block placed at indices `(dim-4, dim-4)` through
   * `(dim-1, dim-1)`.
   */
  void setBottomRightCorner(const Matrix4& block) {
    const std::int64_t offset = dim_ - 4;
    for (std::int64_t i = 0; i < 4; ++i) {
      for (std::int64_t j = 0; j < 4; ++j) {
        (*this)(offset + i, offset + j) =
            block(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
      }
    }
  }

  /**
   * @brief Copies a dynamic block into the bottom-right corner.
   * @param block Source block placed at indices `(dim-block.dim, ...)` through
   * `(dim-1, dim-1)`.
   */
  void setBottomRightCorner(const DynamicMatrix& block) {
    const std::int64_t offset = dim_ - block.dim_;
    for (std::int64_t i = 0; i < block.dim_; ++i) {
      for (std::int64_t j = 0; j < block.dim_; ++j) {
        (*this)(offset + i, offset + j) = block(i, j);
      }
    }
  }

  /**
   * @brief Replaces this matrix with its conjugate transpose in place.
   */
  void adjointInPlace() {
    for (std::int64_t i = 0; i < dim_; ++i) {
      for (std::int64_t j = i + 1; j < dim_; ++j) {
        const Complex tmp = (*this)(i, j);
        (*this)(i, j) = std::conj((*this)(j, i));
        (*this)(j, i) = std::conj(tmp);
      }
      (*this)(i, i) = std::conj((*this)(i, i));
    }
  }

  /**
   * @brief Checks approximate equality against a fixed 4x4 matrix.
   *
   * Returns false if this matrix is not 4x4.
   *
   * @param other Fixed-size matrix to compare against.
   * @param tol Maximum allowed absolute difference per entry.
   * @return True if dimensions match and every entry differs by at most @p tol.
   */
  [[nodiscard]] bool isApprox(const Matrix4& other,
                              double tol = UNITARY_MATRIX_TOLERANCE) const {
    if (dim_ != 4) {
      return false;
    }
    for (std::int64_t i = 0; i < 4; ++i) {
      for (std::int64_t j = 0; j < 4; ++j) {
        if (std::abs((*this)(i, j) - other(static_cast<std::size_t>(i),
                                           static_cast<std::size_t>(j))) >
            tol) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * @brief Checks approximate equality against another dynamic matrix.
   *
   * Returns false if the dimensions differ.
   *
   * @param other Matrix to compare against.
   * @param tol Maximum allowed absolute difference per entry.
   * @return True if dimensions match and every entry differs by at most @p tol.
   */
  [[nodiscard]] bool isApprox(const DynamicMatrix& other,
                              double tol = UNITARY_MATRIX_TOLERANCE) const {
    if (dim_ != other.dim_) {
      return false;
    }
    for (std::size_t idx = 0; idx < data_.size(); ++idx) {
      if (std::abs(data_[idx] - other.data_[idx]) > tol) {
        return false;
      }
    }
    return true;
  }

private:
  /// Side length of the square matrix.
  std::int64_t dim_ = 0;
  /// Flat row-major storage of all matrix entries.
  llvm::SmallVector<Complex> data_;
};

/**
 * @brief Returns the determinant of a 4x4 matrix.
 *
 * Computed via Laplace expansion along the first row.
 */
inline Complex Matrix4::determinant() const {
  auto det3 = [](Complex m00, Complex m01, Complex m02, Complex m10,
                 Complex m11, Complex m12, Complex m20, Complex m21,
                 Complex m22) {
    return m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) +
           m02 * (m10 * m21 - m11 * m20);
  };
  return (*this)(0, 0) * det3((*this)(1, 1), (*this)(1, 2), (*this)(1, 3),
                              (*this)(2, 1), (*this)(2, 2), (*this)(2, 3),
                              (*this)(3, 1), (*this)(3, 2), (*this)(3, 3)) -
         (*this)(0, 1) * det3((*this)(1, 0), (*this)(1, 2), (*this)(1, 3),
                              (*this)(2, 0), (*this)(2, 2), (*this)(2, 3),
                              (*this)(3, 0), (*this)(3, 2), (*this)(3, 3)) +
         (*this)(0, 2) * det3((*this)(1, 0), (*this)(1, 1), (*this)(1, 3),
                              (*this)(2, 0), (*this)(2, 1), (*this)(2, 3),
                              (*this)(3, 0), (*this)(3, 1), (*this)(3, 3)) -
         (*this)(0, 3) * det3((*this)(1, 0), (*this)(1, 1), (*this)(1, 2),
                              (*this)(2, 0), (*this)(2, 1), (*this)(2, 2),
                              (*this)(3, 0), (*this)(3, 1), (*this)(3, 2));
}

/**
 * @brief Type trait that identifies supported unitary matrix types.
 * @tparam T Candidate type.
 */
template <typename T> struct IsUnitaryMatrix : std::false_type {};

/// @brief Specialization for `Matrix1x1`.
template <> struct IsUnitaryMatrix<Matrix1x1> : std::true_type {};
/// @brief Specialization for `Matrix2`.
template <> struct IsUnitaryMatrix<Matrix2> : std::true_type {};
/// @brief Specialization for `Matrix4`.
template <> struct IsUnitaryMatrix<Matrix4> : std::true_type {};
/// @brief Specialization for `DynamicMatrix`.
template <> struct IsUnitaryMatrix<DynamicMatrix> : std::true_type {};

/**
 * @brief Convenience variable template for `IsUnitaryMatrix<T>::value`.
 * @tparam T Candidate type.
 */
template <typename T>
inline constexpr bool IS_UNITARY_MATRIX_V = IsUnitaryMatrix<T>::value;

namespace detail {

/**
 * @brief Copies a 1x1 matrix into a dynamic matrix.
 * @param out Destination matrix, resized to 1x1.
 * @param src Source matrix.
 */
inline void copyInto(DynamicMatrix& out, const Matrix1x1& src) {
  out = DynamicMatrix(1);
  out(0, 0) = src(0, 0);
}

/**
 * @brief Copies a 2x2 matrix into a dynamic matrix.
 * @param out Destination matrix, resized to 2x2.
 * @param src Source matrix.
 */
inline void copyInto(DynamicMatrix& out, const Matrix2& src) {
  out = DynamicMatrix(2);
  for (std::size_t i = 0; i < Matrix2::K_ROWS; ++i) {
    for (std::size_t j = 0; j < Matrix2::K_COLS; ++j) {
      out(static_cast<std::int64_t>(i), static_cast<std::int64_t>(j)) =
          src(i, j);
    }
  }
}

/**
 * @brief Copies a 4x4 matrix into a dynamic matrix.
 * @param out Destination matrix, resized to 4x4.
 * @param src Source matrix.
 */
inline void copyInto(DynamicMatrix& out, const Matrix4& src) {
  out = DynamicMatrix(4);
  for (std::size_t i = 0; i < Matrix4::K_ROWS; ++i) {
    for (std::size_t j = 0; j < Matrix4::K_COLS; ++j) {
      out(static_cast<std::int64_t>(i), static_cast<std::int64_t>(j)) =
          src(i, j);
    }
  }
}

/**
 * @brief Copies a dynamic matrix into another dynamic matrix.
 * @param out Destination matrix.
 * @param src Source matrix.
 */
inline void copyInto(DynamicMatrix& out, const DynamicMatrix& src) {
  out = src;
}

/**
 * @brief Promotes a fixed- or dynamic-size matrix into a dynamic output matrix.
 * @tparam Src Source matrix type.
 * @param out Destination matrix.
 * @param src Source matrix.
 * @return Always `true` on success.
 */
template <typename Src>
inline bool assignToDynamic(DynamicMatrix& out, const Src& src) {
  copyInto(out, src);
  return true;
}

} // namespace detail

} // namespace mlir::qco
