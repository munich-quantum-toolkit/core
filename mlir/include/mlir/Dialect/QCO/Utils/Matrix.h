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

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Support/LLVM.h>

#include <array>
#include <cmath>
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

class DynamicMatrix;
struct Matrix4x4;
struct SymmetricEigen4;

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
  [[nodiscard]] static constexpr Matrix1x1 fromElements(Complex m00) {
    return {m00};
  }

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
   * @brief Element-wise scaling by a complex scalar.
   * @param scalar Factor applied to the matrix entry.
   * @return Scaled copy of this matrix.
   */
  [[nodiscard]] Matrix1x1 operator*(const Complex& scalar) const;

  /**
   * @brief Element-wise in-place scaling by a complex scalar.
   * @param scalar Factor applied to the matrix entry.
   * @return Reference to this matrix.
   */
  Matrix1x1& operator*=(const Complex& scalar);

  /**
   * @brief Returns the conjugate transpose (adjoint) of this matrix.
   * @return Adjoint matrix `A^\dagger`.
   */
  [[nodiscard]] Matrix1x1 adjoint() const;

  /**
   * @brief Checks approximate equality using an absolute tolerance.
   * @param other Matrix to compare against.
   * @param tol Maximum allowed complex modulus of the entry difference.
   * @return True if the difference is within @p tol.
   */
  [[nodiscard]] bool isApprox(const Matrix1x1& other,
                              double tol = MATRIX_TOLERANCE) const;

  /**
   * @brief Replaces this matrix with a copy of a 1x1 dynamic matrix.
   *
   * @param src Source matrix.
   * @return `true` when @p src is 1x1.
   */
  [[nodiscard]] bool assignFrom(const DynamicMatrix& src);
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
  [[nodiscard]] static constexpr Matrix2x2 fromElements(const Complex& m00,
                                                        const Complex& m01,
                                                        const Complex& m10,
                                                        const Complex& m11) {
    return {{m00, m01, m10, m11}};
  }

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
   * @brief Premultiplies by a matrix: `*this = lhs * *this`.
   * @param lhs Left-hand factor.
   */
  void premultiplyBy(const Matrix2x2& lhs);

  /**
   * @brief Element-wise scaling by a complex scalar.
   * @param scalar Factor applied to every matrix entry.
   * @return Scaled copy of this matrix.
   */
  [[nodiscard]] Matrix2x2 operator*(const Complex& scalar) const;

  /**
   * @brief Element-wise in-place scaling by a complex scalar.
   * @param scalar Factor applied to every matrix entry.
   * @return Reference to this matrix.
   */
  Matrix2x2& operator*=(const Complex& scalar);

  /**
   * @brief Returns the conjugate transpose (adjoint) of this matrix.
   * @return Adjoint matrix `A^\dagger`.
   */
  [[nodiscard]] Matrix2x2 adjoint() const;

  /**
   * @brief Returns the (non-conjugate) transpose of this matrix.
   * @return Transposed matrix `A^T`.
   */
  [[nodiscard]] Matrix2x2 transpose() const;

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
   * @brief Checks whether this matrix is approximately the identity.
   * @param tol Maximum allowed complex modulus of each entry difference.
   * @return True if every entry is within @p tol of the identity.
   */
  [[nodiscard]] bool isIdentity(double tol = MATRIX_TOLERANCE) const;

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

  /**
   * @brief Replaces this matrix with a copy of a 2x2 dynamic matrix.
   *
   * @param src Source matrix.
   * @return `true` when @p src is 2x2.
   */
  [[nodiscard]] bool assignFrom(const DynamicMatrix& src);

  /**
   * @brief Embed this single-qubit matrix into an @p numQubits-qubit Hilbert
   * space.
   *
   * Wire @p qubitIndex uses the same MSB-first convention as @ref
   * Matrix4x4::kron (high bit first operand, low bit second). For each basis
   * pair whose untouched wires match, copies this matrix at the target qubit's
   * row/column bits.
   *
   * @param numQubits Number of qubits in the target Hilbert space.
   * @param qubitIndex Wire index to act on.
   * @return Embedded unitary as a dynamic matrix.
   */
  [[nodiscard]] DynamicMatrix embedInNqubit(std::size_t numQubits,
                                            std::size_t qubitIndex) const;

  /**
   * @brief Embed this single-qubit matrix into a two-qubit Hilbert space.
   *
   * @param qubitIndex Wire index (`0` = high bit / MSB, `1` = low bit).
   * @return The `4x4` embedded unitary.
   */
  [[nodiscard]] Matrix4x4 embedInTwoQubit(std::size_t qubitIndex) const;
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
  [[nodiscard]] static constexpr Matrix4x4
  fromElements(const Complex& m00, const Complex& m01, const Complex& m02,
               const Complex& m03, const Complex& m10, const Complex& m11,
               const Complex& m12, const Complex& m13, const Complex& m20,
               const Complex& m21, const Complex& m22, const Complex& m23,
               const Complex& m30, const Complex& m31, const Complex& m32,
               const Complex& m33) {
    return {{m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30,
             m31, m32, m33}};
  }

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
   * @brief Premultiplies by a matrix: `*this = lhs * *this`.
   * @param lhs Left-hand factor.
   */
  void premultiplyBy(const Matrix4x4& lhs);

  /**
   * @brief Element-wise scaling by a complex scalar.
   * @param scalar Factor applied to every matrix entry.
   * @return Scaled copy of this matrix.
   */
  [[nodiscard]] Matrix4x4 operator*(const Complex& scalar) const;

  /**
   * @brief Element-wise in-place scaling by a complex scalar.
   * @param scalar Factor applied to every matrix entry.
   * @return Reference to this matrix.
   */
  Matrix4x4& operator*=(const Complex& scalar);

  /**
   * @brief Returns the conjugate transpose (adjoint) of this matrix.
   * @return Adjoint matrix `A^\dagger`.
   */
  [[nodiscard]] Matrix4x4 adjoint() const;

  /**
   * @brief Returns the (non-conjugate) transpose of this matrix.
   * @return Transposed matrix `A^T`.
   */
  [[nodiscard]] Matrix4x4 transpose() const;

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
   * @brief Checks whether this matrix is approximately the identity.
   * @param tol Maximum allowed complex modulus of each entry difference.
   * @return True if every entry is within @p tol of the identity.
   */
  [[nodiscard]] bool isIdentity(double tol = MATRIX_TOLERANCE) const;

  /**
   * @brief Returns the four diagonal entries `(m00, m11, m22, m33)`.
   * @return Array of diagonal entries.
   */
  [[nodiscard]] std::array<Complex, K_ROWS> diagonal() const;

  /**
   * @brief Builds a diagonal matrix from four diagonal entries.
   * @param diagonalEntries Diagonal entries `(m00, m11, m22, m33)`; must have
   *        length `K_ROWS`.
   * @return Diagonal matrix with the given entries.
   */
  [[nodiscard]] static Matrix4x4
  fromDiagonal(ArrayRef<Complex> diagonalEntries);

  /**
   * @brief Kronecker product `lhs (x) rhs` of two single-qubit matrices.
   *
   * Uses the computational-basis bit order where the first operand labels the
   * high bit, matching `UnitaryOpInterface::getUnitaryMatrix4x4`.
   *
   * @param lhs Left factor (acts on the high bit / qubit 0).
   * @param rhs Right factor (acts on the low bit / qubit 1).
   * @return The `4x4` Kronecker product.
   */
  [[nodiscard]] static Matrix4x4 kron(const Matrix2x2& lhs,
                                      const Matrix2x2& rhs);

  /**
   * @brief Returns the entries of column @p col, top to bottom.
   * @param col Column index in `[0, K_COLS)`.
   * @return Array of the four column entries.
   */
  [[nodiscard]] std::array<Complex, K_ROWS> column(std::size_t col) const;

  /**
   * @brief Overwrites column @p col with @p values.
   * @param col Column index in `[0, K_COLS)`.
   * @param values New column entries, top to bottom; must have length `K_ROWS`.
   */
  void setColumn(std::size_t col, ArrayRef<Complex> values);

  /**
   * @brief Returns the entries of row @p row, left to right.
   * @param row Row index in `[0, K_ROWS)`.
   * @return View over the four row entries.
   */
  [[nodiscard]] ArrayRef<const Complex> row(std::size_t row) const;

  /**
   * @brief Overwrites row @p row with @p values.
   * @param row Row index in `[0, K_ROWS)`.
   * @param values New row entries, left to right; must have length `K_COLS`.
   */
  void setRow(std::size_t row, ArrayRef<Complex> values);

  /**
   * @brief Returns the element-wise real parts in row-major order.
   * @return Real parts of all entries.
   */
  [[nodiscard]] std::array<double, K_SIZE_AT_COMPILE_TIME> realPart() const;

  /**
   * @brief Returns the element-wise imaginary parts in row-major order.
   * @return Imaginary parts of all entries.
   */
  [[nodiscard]] std::array<double, K_SIZE_AT_COMPILE_TIME> imagPart() const;

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

  /**
   * @brief Replaces this matrix with a copy of a 4x4 dynamic matrix.
   *
   * @param src Source matrix.
   * @return `true` when @p src is 4x4.
   */
  [[nodiscard]] bool assignFrom(const DynamicMatrix& src);

  /**
   * @brief Embed this two-qubit matrix into an @p numQubits-qubit Hilbert
   * space.
   *
   * Operand 0 labels the high bit of the pair and acts on @p q0Index; operand 1
   * labels the low bit and acts on @p q1Index. For each basis pair whose other
   * wires match, copies this matrix at the packed two-qubit row/column indices.
   *
   * @param numQubits Number of qubits in the target Hilbert space.
   * @param q0Index Wire index of operand 0.
   * @param q1Index Wire index of operand 1.
   * @return Embedded unitary as a dynamic matrix.
   */
  [[nodiscard]] DynamicMatrix embedInNqubit(std::size_t numQubits,
                                            std::size_t q0Index,
                                            std::size_t q1Index) const;

  /**
   * @brief Reorder this matrix to act on qubits `{0, 1}`.
   *
   * @param q0Index Wire index of operand 0; @p q1Index wire index of operand 1.
   * @return Reordered copy of this matrix.
   */
  [[nodiscard]] Matrix4x4 reorderForQubits(std::size_t q0Index,
                                           std::size_t q1Index) const;

  /**
   * @brief Computes the eigendecomposition of this real symmetric matrix.
   *
   * @copydoc Matrix4x4::symmetricEigen4(const std::array<double, 16>&)
   *
   * @pre Entries are real (imaginary parts must be negligible). The real parts
   * must form a symmetric matrix; imaginary parts are ignored.
   */
  [[nodiscard]] SymmetricEigen4 symmetricEigen4() const;

  /**
   * @brief Computes the eigendecomposition of a real symmetric `4x4` matrix.
   *
   * Uses Householder tridiagonalization (EISPACK `tred2`) followed by implicit
   * QL iteration (`tql2`) on the tridiagonal form.
   *
   * @pre @p symmetric is real and symmetric: `symmetric[i,j] == symmetric[j,i]`
   * for all `i, j`. Only the lower triangle (including the diagonal) is read,
   * but supplying a non-symmetric matrix yields undefined numerical results.
   *
   * @param symmetric Row-major real symmetric `4x4` matrix.
   * @return Ascending eigenvalues and matching eigenvectors (as columns).
   */
  [[nodiscard]] static SymmetricEigen4
  symmetricEigen4(const std::array<double, 16>& symmetric);
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

  /**
   * @brief Creates a dynamic matrix from a fixed 2x2 matrix.
   * @param src Source matrix.
   */
  explicit DynamicMatrix(const Matrix2x2& src);

  /**
   * @brief Creates a dynamic matrix from a fixed 4x4 matrix.
   * @param src Source matrix.
   */
  explicit DynamicMatrix(const Matrix4x4& src);

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
   * @brief Creates a dynamic matrix holding the adjoint of a 2x2 matrix.
   * @param src Source matrix.
   * @return Adjoint matrix `src^\dagger`.
   */
  [[nodiscard]] static DynamicMatrix fromAdjoint(const Matrix2x2& src);

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
   * @brief Checks approximate equality against a fixed 1x1 matrix.
   *
   * Returns false if this matrix is not 1x1.
   *
   * @param other Fixed-size matrix to compare against.
   * @param tol Maximum allowed complex modulus of the entry difference.
   * @return True if dimensions match and the entry differs by at most @p tol.
   */
  [[nodiscard]] bool isApprox(const Matrix1x1& other,
                              double tol = MATRIX_TOLERANCE) const;

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

  /**
   * @brief Returns the trace of this matrix.
   * @return Sum of diagonal entries.
   */
  [[nodiscard]] Complex trace() const;

  /**
   * @brief Matrix product `*this * rhs`.
   * @param rhs Right-hand factor.
   * @return Product of the two matrices.
   */
  [[nodiscard]] DynamicMatrix operator*(const DynamicMatrix& rhs) const;

  /**
   * @brief Element-wise scaling by a complex scalar.
   * @param scalar Factor applied to every matrix entry.
   * @return Scaled copy of this matrix.
   */
  [[nodiscard]] DynamicMatrix operator*(const Complex& scalar) const;

  /**
   * @brief Element-wise in-place scaling by a complex scalar.
   * @param scalar Factor applied to every matrix entry.
   * @return Reference to this matrix.
   */
  DynamicMatrix& operator*=(const Complex& scalar);

  /**
   * @brief Checks whether this matrix is approximately the identity.
   * @param tol Maximum allowed complex modulus of each off-diagonal entry and
   * each diagonal deviation from one.
   * @return True when the matrix is close to the identity.
   */
  [[nodiscard]] bool isIdentity(double tol = MATRIX_TOLERANCE) const;

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

/**
 * @brief Checks whether two unitaries are equal up to a global phase.
 *
 * Uses `trace(rhs.adjoint() * lhs)` to infer a unit-modulus phase factor, then
 * compares `lhs` to `factor * rhs` with @ref isApprox. A near-zero overlap
 * (`|trace| <= tol`) is treated as not equivalent to avoid dividing by a tiny
 * number.
 *
 * @tparam Matrix Any supported fixed- or dynamic-size matrix type.
 * @param lhs Left-hand unitary.
 * @param rhs Right-hand unitary.
 * @param tol Absolute tolerance for overlap and entry-wise comparison.
 * @return True when @p lhs and @p rhs differ only by a global phase.
 */
template <typename Matrix>
  requires is_supported_matrix_v<Matrix>
[[nodiscard]] bool isEquivalentUpToGlobalPhase(const Matrix& lhs,
                                               const Matrix& rhs,
                                               double tol = MATRIX_TOLERANCE) {
  const auto overlap = (rhs.adjoint() * lhs).trace();
  if (std::abs(overlap) <= tol) {
    return false;
  }
  const auto factor = overlap / std::abs(overlap);
  return lhs.isApprox(factor * rhs, tol);
}

/// Scalar-on-the-left multiply `scalar * matrix` (commutes with the member
/// `matrix * scalar`). Provided so generic code can scale a matrix from
/// either side.
[[nodiscard]] Matrix2x2 operator*(const Complex& scalar,
                                  const Matrix2x2& matrix);
/// @copydoc operator*(const Complex&, const Matrix2x2&)
[[nodiscard]] Matrix4x4 operator*(const Complex& scalar,
                                  const Matrix4x4& matrix);
/// @copydoc operator*(const Complex&, const Matrix2x2&)
[[nodiscard]] DynamicMatrix operator*(const Complex& scalar,
                                      const DynamicMatrix& matrix);

/**
 * @brief Eigenvalues and eigenvectors of a real symmetric `4x4` matrix.
 *
 * `eigenvalues` are sorted ascending and `eigenvectors` holds the
 * corresponding orthonormal eigenvectors as columns (column `j` is the
 * eigenvector for `eigenvalues[j]`).
 */
struct SymmetricEigen4 {
  /// Eigenvalues in ascending order.
  std::array<double, 4> eigenvalues{};
  /// Orthonormal eigenvectors as columns (column `j` matches `eigenvalues[j]`).
  Matrix4x4 eigenvectors{};
};

} // namespace mlir::qco
