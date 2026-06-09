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

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace mlir::qco {

/// Complex scalar type used for matrix entries.
using Complex = std::complex<double>;

/// Default absolute tolerance for matrix comparisons.
inline constexpr double MATRIX_TOLERANCE = 1e-14;

/**
 * @brief 1x1 matrix for global-phase gates.
 *
 * Wraps a single complex scalar. Used by operations such as `GPhaseOp` whose
 * unitary is a global phase factor.
 */
struct Matrix1x1 {
  /// The sole matrix entry.
  Complex value{0.0, 0.0};

  /**
   * @brief Constructs a matrix from its single entry.
   * @param m00 Element at row 0, column 0.
   * @return A new `Matrix1x1` with the given element.
   */
  [[nodiscard]] static Matrix1x1 fromElements(Complex m00);

  /**
   * @brief Mutable element access with `(row, col)` indexing.
   * @param row Row index (must be `0`).
   * @param col Column index (must be `0`).
   * @return Reference to the sole matrix entry.
   */
  [[nodiscard]] Complex& operator()(std::size_t row, std::size_t col);

  /**
   * @brief Const element access with `(row, col)` indexing.
   * @param row Row index (must be `0`).
   * @param col Column index (must be `0`).
   * @return Copy of the sole matrix entry.
   */
  [[nodiscard]] Complex operator()(std::size_t row, std::size_t col) const;

  /**
   * @brief Checks approximate equality using an absolute tolerance.
   * @param other Matrix to compare against.
   * @param tol Maximum allowed complex modulus of the entry difference.
   * @return True if the difference is within @p tol.
   */
  [[nodiscard]] bool isApprox(const Matrix1x1& other,
                              double tol = MATRIX_TOLERANCE) const;
};

/**
 * @brief Fixed-size 2x2 matrix in row-major layout.
 *
 * Used to represent single-qubit unitaries. Elements are stored in a flat array
 * with index `(row * K_COLS) + col`.
 */
struct Matrix2x2 {
  /// Number of rows.
  static constexpr std::size_t K_ROWS = 2;
  /// Number of columns.
  static constexpr std::size_t K_COLS = 2;
  /// Total number of stored elements.
  static constexpr std::size_t K_SIZE_AT_COMPILE_TIME = 4;

  /// Flat row-major storage of all matrix entries.
  std::array<Complex, K_SIZE_AT_COMPILE_TIME> data{};

  /**
   * @brief Constructs a matrix from its four row-major entries.
   * @param m00 Element at row 0, column 0.
   * @param m01 Element at row 0, column 1.
   * @param m10 Element at row 1, column 0.
   * @param m11 Element at row 1, column 1.
   * @return A new `Matrix2x2` with the given elements.
   */
  [[nodiscard]] static Matrix2x2 fromElements(Complex m00, Complex m01,
                                              Complex m10, Complex m11);

  /**
   * @brief Returns the 2x2 identity matrix.
   * @return Identity matrix `[[1, 0], [0, 1]]`.
   */
  [[nodiscard]] static constexpr Matrix2x2 identity() { return {{1, 0, 0, 1}}; }

  /**
   * @brief Mutable element access.
   * @param row Row index in `[0, K_ROWS)`.
   * @param col Column index in `[0, K_COLS)`.
   * @return Reference to the element at `(row, col)`.
   */
  [[nodiscard]] Complex& operator()(std::size_t row, std::size_t col);

  /**
   * @brief Const element access.
   * @param row Row index in `[0, K_ROWS)`.
   * @param col Column index in `[0, K_COLS)`.
   * @return Copy of the element at `(row, col)`.
   */
  [[nodiscard]] Complex operator()(std::size_t row, std::size_t col) const;

  /**
   * @brief Matrix product `*this * rhs`.
   * @param rhs Right-hand factor.
   * @return Product of the two matrices.
   */
  [[nodiscard]] Matrix2x2 operator*(const Matrix2x2& rhs) const;

  /**
   * @brief Returns the conjugate transpose (adjoint) of this matrix.
   * @return Adjoint matrix `A^\dagger`.
   */
  [[nodiscard]] Matrix2x2 adjoint() const;

  /**
   * @brief Returns the trace of this matrix.
   * @return Sum of diagonal entries.
   */
  [[nodiscard]] Complex trace() const;

  /**
   * @brief Returns the determinant of this matrix.
   * @return Complex determinant `ad - bc`.
   */
  [[nodiscard]] Complex determinant() const;

  /**
   * @brief Checks approximate equality using an absolute entry-wise tolerance.
   *
   * For each entry `i`, the comparison uses `|a_i - b_i| <= tol`, where the
   * absolute value is the complex modulus.
   *
   * @param other Matrix to compare against.
   * @param tol Maximum allowed complex modulus of each entry difference.
   * @return True if every entry differs by at most @p tol.
   */
  [[nodiscard]] bool isApprox(const Matrix2x2& other,
                              double tol = MATRIX_TOLERANCE) const;
};

/**
 * @brief Fixed-size 4x4 matrix in row-major layout.
 *
 * Used to represent two-qubit gate unitaries. Elements are stored in a flat
 * array with index `(row * K_COLS) + col`.
 */
struct Matrix4x4 {
  /// Number of rows.
  static constexpr std::size_t K_ROWS = 4;
  /// Number of columns.
  static constexpr std::size_t K_COLS = 4;
  /// Total number of stored elements.
  static constexpr std::size_t K_SIZE_AT_COMPILE_TIME = 16;

  /// Flat row-major storage of all matrix entries.
  std::array<Complex, K_SIZE_AT_COMPILE_TIME> data{};

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
   * @return A new `Matrix4x4` with the given elements.
   */
  [[nodiscard]] static Matrix4x4
  fromElements(Complex m00, Complex m01, Complex m02, Complex m03, Complex m10,
               Complex m11, Complex m12, Complex m13, Complex m20, Complex m21,
               Complex m22, Complex m23, Complex m30, Complex m31, Complex m32,
               Complex m33);

  /**
   * @brief Returns the 4x4 identity matrix.
   * @return Identity matrix with ones on the diagonal.
   */
  [[nodiscard]] static constexpr Matrix4x4 identity() {
    return {{1, 0, 0, 0,   // row 0
             0, 1, 0, 0,   // row 1
             0, 0, 1, 0,   // row 2
             0, 0, 0, 1}}; // row 3
  }

  /**
   * @brief Mutable element access.
   * @param row Row index in `[0, K_ROWS)`.
   * @param col Column index in `[0, K_COLS)`.
   * @return Reference to the element at `(row, col)`.
   */
  [[nodiscard]] Complex& operator()(std::size_t row, std::size_t col);

  /**
   * @brief Const element access.
   * @param row Row index in `[0, K_ROWS)`.
   * @param col Column index in `[0, K_COLS)`.
   * @return Copy of the element at `(row, col)`.
   */
  [[nodiscard]] Complex operator()(std::size_t row, std::size_t col) const;

  /**
   * @brief Matrix product `*this * rhs`.
   * @param rhs Right-hand factor.
   * @return Product of the two matrices.
   */
  [[nodiscard]] Matrix4x4 operator*(const Matrix4x4& rhs) const;

  /**
   * @brief Returns the conjugate transpose (adjoint) of this matrix.
   * @return Adjoint matrix `A^\dagger`.
   */
  [[nodiscard]] Matrix4x4 adjoint() const;

  /**
   * @brief Returns the trace of this matrix.
   * @return Sum of diagonal entries.
   */
  [[nodiscard]] Complex trace() const;

  /**
   * @brief Returns the determinant of this matrix.
   * @return Complex determinant computed via Laplace expansion.
   */
  [[nodiscard]] Complex determinant() const;

  /**
   * @brief Checks approximate equality using an absolute entry-wise tolerance.
   *
   * For each entry `i`, the comparison uses `|a_i - b_i| <= tol`, where the
   * absolute value is the complex modulus.
   *
   * @param other Matrix to compare against.
   * @param tol Maximum allowed complex modulus of each entry difference.
   * @return True if every entry differs by at most @p tol.
   */
  [[nodiscard]] bool isApprox(const Matrix4x4& other,
                              double tol = MATRIX_TOLERANCE) const;
};

/**
 * @brief Square matrix with runtime dimension.
 *
 * Used when the Hilbert-space dimension depends on the operation, for example,
 * in controlled gates (`CtrlOp`) and inverses (`InvOp`). Storage is row-major
 * and held behind a private implementation pointer.
 */
class DynamicMatrix {
public:
  /// Creates an empty 0x0 matrix.
  DynamicMatrix();

  /**
   * @brief Creates a zero-initialized square matrix.
   * @param dim Side length of the square matrix.
   */
  explicit DynamicMatrix(std::int64_t dim);

  /// Copy constructor.
  DynamicMatrix(const DynamicMatrix& other);
  /// Move constructor.
  DynamicMatrix(DynamicMatrix&& other) noexcept;
  /// Copy assignment.
  DynamicMatrix& operator=(const DynamicMatrix& other);
  /// Move assignment.
  DynamicMatrix& operator=(DynamicMatrix&& other) noexcept;
  /// Destructor.
  ~DynamicMatrix();

  /**
   * @brief Returns a square identity matrix of the given dimension.
   * @param dim Side length of the identity matrix.
   * @return Identity matrix with ones on the diagonal.
   */
  [[nodiscard]] static DynamicMatrix identity(std::int64_t dim);

  /**
   * @brief Returns the number of rows.
   * @return Matrix dimension.
   */
  [[nodiscard]] std::int64_t rows() const;

  /**
   * @brief Returns the number of columns.
   * @return Matrix dimension.
   */
  [[nodiscard]] std::int64_t cols() const;

  /**
   * @brief Mutable element access.
   * @param row Row index in `[0, dim)`.
   * @param col Column index in `[0, dim)`.
   * @return Reference to the element at `(row, col)`.
   */
  [[nodiscard]] Complex& operator()(std::int64_t row, std::int64_t col);

  /**
   * @brief Const element access.
   * @param row Row index in `[0, dim)`.
   * @param col Column index in `[0, dim)`.
   * @return Copy of the element at `(row, col)`.
   */
  [[nodiscard]] Complex operator()(std::int64_t row, std::int64_t col) const;

  /**
   * @brief Copies a 2x2 block into the bottom-right corner.
   * @param block Source block placed at indices `(dim-2, dim-2)` through
   * `(dim-1, dim-1)`.
   */
  void setBottomRightCorner(const Matrix2x2& block);

  /**
   * @brief Copies a 4x4 block into the bottom-right corner.
   * @param block Source block placed at indices `(dim-4, dim-4)` through
   * `(dim-1, dim-1)`.
   */
  void setBottomRightCorner(const Matrix4x4& block);

  /**
   * @brief Copies a dynamic block into the bottom-right corner.
   * @param block Source block placed at indices `(dim - block.rows(), ...)`
   * through
   * `(dim-1, dim-1)`.
   */
  void setBottomRightCorner(const DynamicMatrix& block);

  /**
   * @brief Returns the conjugate transpose (adjoint) of this matrix.
   * @return Adjoint matrix `A^\dagger`.
   */
  [[nodiscard]] DynamicMatrix adjoint() const;

  /**
   * @brief Replaces this matrix with a copy of a 1x1 matrix.
   * @param src Source matrix.
   */
  void assignFrom(const Matrix1x1& src);

  /**
   * @brief Replaces this matrix with a copy of a 2x2 matrix.
   * @param src Source matrix.
   */
  void assignFrom(const Matrix2x2& src);

  /**
   * @brief Replaces this matrix with a copy of a 4x4 matrix.
   * @param src Source matrix.
   */
  void assignFrom(const Matrix4x4& src);

  /**
   * @brief Replaces this matrix with a copy of another dynamic matrix.
   * @param src Source matrix.
   */
  void assignFrom(const DynamicMatrix& src);

  /**
   * @brief Checks approximate equality against a fixed 2x2 matrix.
   *
   * Returns false if this matrix is not 2x2.
   *
   * @param other Fixed-size matrix to compare against.
   * @param tol Maximum allowed complex modulus of each entry difference.
   * @return True if dimensions match and every entry differs by at most @p tol.
   */
  [[nodiscard]] bool isApprox(const Matrix2x2& other,
                              double tol = MATRIX_TOLERANCE) const;

  /**
   * @brief Checks approximate equality against a fixed 4x4 matrix.
   *
   * Returns false if this matrix is not 4x4.
   *
   * @param other Fixed-size matrix to compare against.
   * @param tol Maximum allowed complex modulus of each entry difference.
   * @return True if dimensions match and every entry differs by at most @p tol.
   */
  [[nodiscard]] bool isApprox(const Matrix4x4& other,
                              double tol = MATRIX_TOLERANCE) const;

  /**
   * @brief Checks approximate equality against another dynamic matrix.
   *
   * Returns false if the dimensions differ.
   *
   * @param other Matrix to compare against.
   * @param tol Maximum allowed complex modulus of each entry difference.
   * @return True if dimensions match and every entry differs by at most @p tol.
   */
  [[nodiscard]] bool isApprox(const DynamicMatrix& other,
                              double tol = MATRIX_TOLERANCE) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Type trait for the four supported matrix types.
 *
 * True for @ref Matrix1x1, @ref Matrix2x2, @ref Matrix4x4, and @ref
 * DynamicMatrix.
 *
 * @tparam T Candidate type.
 */
template <typename T>
inline constexpr bool
    is_supported_matrix_v = // NOLINT(readability-identifier-naming)
    std::disjunction_v<std::is_same<T, Matrix1x1>, std::is_same<T, Matrix2x2>,
                       std::is_same<T, Matrix4x4>,
                       std::is_same<T, DynamicMatrix>>;
} // namespace mlir::qco
