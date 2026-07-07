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

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <utility>

using namespace mlir::qco;
using namespace std::complex_literals;

static_assert(SupportedMatrix<Matrix1x1>);
static_assert(SupportedMatrix<Matrix2x2>);
static_assert(SupportedMatrix<Matrix4x4>);
static_assert(SupportedMatrix<DynamicMatrix>);
static_assert(!SupportedMatrix<int>);

[[nodiscard]] static constexpr Matrix2x2 pauliX() {
  return Matrix2x2::fromElements(0, 1, 1, 0);
}

[[nodiscard]] static constexpr Matrix4x4 swapMatrix() {
  return Matrix4x4::fromElements(1, 0, 0, 0,  // row 0
                                 0, 0, 1, 0,  // row 1
                                 0, 1, 0, 0,  // row 2
                                 0, 0, 0, 1); // row 3
}

[[nodiscard]] static DynamicMatrix cyclicPermutation(const std::int64_t dim) {
  DynamicMatrix matrix(dim);
  for (std::int64_t row = 0; row < dim; ++row) {
    matrix(row, (row + 1) % dim) = 1.0;
  }
  return matrix;
}

[[nodiscard]] static bool
complexesAreApprox(const Complex& lhs, const Complex& rhs, const double tol) {
  return std::abs(lhs - rhs) <= tol;
}

static void verifyMatrix2x2FixedMatchesDynamic() {
  const Matrix2x2 gate =
      Matrix2x2::fromElements(std::exp(1i * 0.3), 0.1, 0.0, std::exp(1i * 1.1));
  const std::optional<EigenDecomposition2x2> fixed = gate.eigenDecomposition();
  const std::optional<EigenDecomposition> dynamic =
      DynamicMatrix(gate).eigenDecomposition();
  ASSERT_TRUE(fixed.has_value());
  ASSERT_TRUE(dynamic.has_value());
  EXPECT_TRUE(DynamicMatrix(fixed->eigenvectors)
                  .isApprox(dynamic->eigenvectors, MATRIX_TOLERANCE));
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_TRUE(complexesAreApprox(fixed->eigenvalues[i],
                                   dynamic->eigenvalues[i], MATRIX_TOLERANCE));
  }
}

static void verifyMatrix4x4FixedMatchesDynamic() {
  const Matrix4x4 gate =
      Matrix4x4::fromDiagonal({std::exp(1i * 0.2), std::exp(1i * 0.5),
                               std::exp(1i * 1.1), std::exp(1i * -0.7)});
  const std::optional<EigenDecomposition4x4> fixed = gate.eigenDecomposition();
  const std::optional<EigenDecomposition> dynamic =
      DynamicMatrix(gate).eigenDecomposition();
  ASSERT_TRUE(fixed.has_value());
  ASSERT_TRUE(dynamic.has_value());
  EXPECT_TRUE(DynamicMatrix(fixed->eigenvectors)
                  .isApprox(dynamic->eigenvectors, MATRIX_TOLERANCE));
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_TRUE(complexesAreApprox(fixed->eigenvalues[i],
                                   dynamic->eigenvalues[i], MATRIX_TOLERANCE));
  }
}

[[nodiscard]] static bool
eigenpairsSatisfy(const DynamicMatrix& matrix,
                  const EigenDecomposition& decomposition,
                  const double tol = MATRIX_TOLERANCE) {
  const std::int64_t dim = matrix.rows();
  for (std::size_t colIdx = 0; colIdx < decomposition.eigenvalues.size();
       ++colIdx) {
    DynamicMatrix eigenvector(dim);
    for (std::int64_t row = 0; row < dim; ++row) {
      eigenvector(row, 0) =
          decomposition.eigenvectors(row, static_cast<std::int64_t>(colIdx));
    }
    const DynamicMatrix image = matrix * eigenvector;
    const Complex lambda = decomposition.eigenvalues[colIdx];
    for (std::int64_t row = 0; row < dim; ++row) {
      if (!complexesAreApprox(image(row, 0), lambda * eigenvector(row, 0),
                              tol)) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] static double matrixFrobeniusNorm(const DynamicMatrix& matrix) {
  double sumSq = 0.0;
  const std::int64_t dim = matrix.rows();
  for (std::int64_t row = 0; row < dim; ++row) {
    for (std::int64_t col = 0; col < dim; ++col) {
      sumSq += std::norm(matrix(row, col));
    }
  }
  return std::sqrt(sumSq);
}

[[nodiscard]] static double
maxEigenpairResidual(const DynamicMatrix& matrix,
                     const EigenDecomposition& decomposition) {
  const std::int64_t dim = matrix.rows();
  double worst = 0.0;
  for (std::size_t colIdx = 0; colIdx < decomposition.eigenvalues.size();
       ++colIdx) {
    DynamicMatrix eigenvector(dim);
    for (std::int64_t row = 0; row < dim; ++row) {
      eigenvector(row, 0) =
          decomposition.eigenvectors(row, static_cast<std::int64_t>(colIdx));
    }
    const DynamicMatrix image = matrix * eigenvector;
    const Complex lambda = decomposition.eigenvalues[colIdx];
    for (std::int64_t row = 0; row < dim; ++row) {
      worst = std::max(worst,
                       std::abs(image(row, 0) - lambda * eigenvector(row, 0)));
    }
  }
  return worst;
}

[[nodiscard]] static double
generalEigenpairTolerance(const DynamicMatrix& matrix) {
  const auto dim =
      static_cast<double>(std::max<std::int64_t>(matrix.rows(), 1));
  const double scale = std::max(1.0, matrixFrobeniusNorm(matrix));
  // EISPACK backward error scales with dimension and ||A||_F.
  return std::max(1e-10, 1e-11 * dim * scale);
}

static void expectEigenDecomposition(const DynamicMatrix& matrix) {
  const std::optional<EigenDecomposition> decomposition =
      matrix.eigenDecomposition();
  ASSERT_TRUE(decomposition.has_value())
      << "eigenDecomposition failed for " << matrix.rows() << "x"
      << matrix.cols();
  ASSERT_EQ(decomposition->eigenvalues.size(),
            static_cast<std::size_t>(matrix.rows()));
  ASSERT_EQ(decomposition->eigenvectors.rows(), matrix.rows());
  ASSERT_EQ(decomposition->eigenvectors.cols(), matrix.cols());
  for (const Complex& eigenvalue : decomposition->eigenvalues) {
    EXPECT_NEAR(std::abs(eigenvalue), 1.0, MATRIX_TOLERANCE);
  }
  EXPECT_TRUE(eigenpairsSatisfy(matrix, *decomposition));
}

static void expectGeneralEigenDecomposition(const DynamicMatrix& matrix) {
  const std::optional<EigenDecomposition> decomposition =
      matrix.eigenDecomposition();
  ASSERT_TRUE(decomposition.has_value())
      << "eigenDecomposition failed for " << matrix.rows() << "x"
      << matrix.cols();
  ASSERT_EQ(decomposition->eigenvalues.size(),
            static_cast<std::size_t>(matrix.rows()));
  ASSERT_EQ(decomposition->eigenvectors.rows(), matrix.rows());
  ASSERT_EQ(decomposition->eigenvectors.cols(), matrix.cols());
  const double tol = generalEigenpairTolerance(matrix);
  const double residual = maxEigenpairResidual(matrix, *decomposition);
  EXPECT_LE(residual, tol) << "max eigenpair residual " << residual
                           << " exceeds tolerance " << tol;

  Complex eigenTrace{0.0, 0.0};
  for (const Complex& eigenvalue : decomposition->eigenvalues) {
    eigenTrace += eigenvalue;
  }
  EXPECT_TRUE(complexesAreApprox(matrix.trace(), eigenTrace, tol));
}

[[nodiscard]] static DynamicMatrix randomComplexMatrix(const std::int64_t dim,
                                                       std::mt19937& rng) {
  std::uniform_real_distribution dist(-3.0, 3.0);
  DynamicMatrix matrix(dim);
  for (std::int64_t row = 0; row < dim; ++row) {
    for (std::int64_t col = 0; col < dim; ++col) {
      matrix(row, col) = Complex{dist(rng), dist(rng)};
    }
  }
  return matrix;
}

[[nodiscard]] static DynamicMatrix diagonalUnitary(const std::int64_t dim) {
  DynamicMatrix matrix(dim);
  for (std::int64_t i = 0; i < dim; ++i) {
    matrix(i, i) = std::exp(1i * (0.1 * static_cast<double>(i)));
  }
  return matrix;
}

[[nodiscard]] static Matrix4x4 nonDiagonalTestMatrix4x4() {
  return Matrix4x4::fromElements(1.0, 0.5, -0.25, 0.0,  // row 0
                                 -1.0, 2.0, 0.75, 0.25, // row 1
                                 0.5, -0.5, 1.5, -1.0,  // row 2
                                 0.0, 1.0, 0.5, -2.0);  // row 3
}

namespace {

struct EigenDecompositionMatrixCase {
  std::string name;
  DynamicMatrix matrix;
};

struct FixedDynamicEigenCase {
  std::string name;
  void (*verify)();
};

class EigenDecompositionMatrixTest
    : public testing::TestWithParam<EigenDecompositionMatrixCase> {};

class EigenDecompositionFixedDynamicTest
    : public testing::TestWithParam<FixedDynamicEigenCase> {};

} // namespace

TEST(UnitaryMatrix1x1, FromElementsAndAccess) {
  Matrix1x1 matrix = Matrix1x1::fromElements(Complex{0.5, 0.5});
  EXPECT_EQ(matrix(0, 0), 0.5 + 0.5i);
  matrix(0, 0) = -1i;
  EXPECT_EQ(matrix.value, -1i);
}

TEST(UnitaryMatrix1x1, ConstElementAccess) {
  const Matrix1x1 matrix = Matrix1x1::fromElements(Complex{0.25, 0.5});
  EXPECT_EQ(matrix(0, 0), (Complex{0.25, 0.5}));
}

TEST(UnitaryMatrix1x1, IsApprox) {
  const Matrix1x1 a = Matrix1x1::fromElements(1.0);
  const Matrix1x1 b = Matrix1x1::fromElements(Complex{1.0, 1e-16});
  EXPECT_TRUE(a.isApprox(b));
  EXPECT_FALSE(a.isApprox(Matrix1x1::fromElements(2.0)));
  EXPECT_TRUE(a.isApprox(Matrix1x1::fromElements(1.1), 0.2));
  EXPECT_EQ((Matrix1x1::fromElements(0.5) * 2.0)(0, 0), 1.0);
  Matrix1x1 scaled = Matrix1x1::fromElements(0.5);
  scaled *= 2.0;
  EXPECT_EQ(scaled(0, 0), 1.0);
}

TEST(UnitaryMatrix1x1, Adjoint) {
  const Matrix1x1 phase = Matrix1x1::fromElements(Complex{0.25, 0.5});
  EXPECT_TRUE(phase.adjoint().isApprox(
      Matrix1x1::fromElements(std::conj(phase.value))));
}

TEST(UnitaryMatrix2x2, IdentityAndAccess) {
  const Matrix2x2 identity = Matrix2x2::identity();
  EXPECT_TRUE(identity.isApprox(Matrix2x2::fromElements(1, 0, 0, 1)));
  EXPECT_EQ(identity(0, 1), 0.0);
  Matrix2x2 mutableMatrix = identity;
  mutableMatrix(1, 1) = 2.0;
  EXPECT_EQ(mutableMatrix(1, 1), 2.0);
}

TEST(UnitaryMatrix2x2, MultiplyAdjointTraceDeterminant) {
  const Matrix2x2 x = pauliX();
  const Matrix2x2 identity = Matrix2x2::identity();

  EXPECT_TRUE((x * x).isApprox(identity));
  EXPECT_TRUE((identity * x).isApprox(x));
  EXPECT_TRUE((x * std::exp(1i * 0.5))
                  .isApprox(Matrix2x2::fromElements(0, std::exp(1i * 0.5),
                                                    std::exp(1i * 0.5), 0)));
  Matrix2x2 scaled = x;
  scaled *= std::exp(1i * 0.5);
  EXPECT_TRUE(scaled.isApprox(x * std::exp(1i * 0.5)));
  EXPECT_TRUE(x.adjoint().isApprox(x));
  EXPECT_EQ(x.trace(), Complex(0.0, 0.0));
  EXPECT_EQ(identity.trace(), Complex(2.0, 0.0));
  EXPECT_EQ(x.determinant(), Complex(-1.0, 0.0));
  EXPECT_EQ(identity.determinant(), Complex(1.0, 0.0));
}

TEST(UnitaryMatrix2x2, PremultiplyBy) {
  const Matrix2x2 x = pauliX();
  const Matrix2x2 y = Matrix2x2::fromElements(1, 0, 0, std::exp(1i * 0.5));
  Matrix2x2 acc = Matrix2x2::identity();
  acc.premultiplyBy(x);
  acc.premultiplyBy(y);
  EXPECT_TRUE(acc.isApprox(y * x));
}

TEST(UnitaryMatrix2x2, IsApprox) {
  const Matrix2x2 a = Matrix2x2::identity();
  Matrix2x2 b = a;
  b(0, 0) += 1e-15;
  EXPECT_TRUE(a.isApprox(b));
  EXPECT_FALSE(a.isApprox(pauliX()));
}

TEST(UnitaryMatrix4x4, IdentityAndAccess) {
  const Matrix4x4 identity = Matrix4x4::identity();
  EXPECT_TRUE(identity.isApprox(Matrix4x4::fromElements(1, 0, 0, 0,    // row 0
                                                        0, 1, 0, 0,    // row 1
                                                        0, 0, 1, 0,    // row 2
                                                        0, 0, 0, 1))); // row 3
  EXPECT_EQ(identity(2, 2), 1.0);
  EXPECT_TRUE((swapMatrix() * 2.0)(0, 0) == 2.0);
}

TEST(UnitaryMatrix4x4, MultiplyAdjointTraceDeterminant) {
  const Matrix4x4 swap = swapMatrix();
  const Matrix4x4 identity = Matrix4x4::identity();

  EXPECT_TRUE((swap * swap).isApprox(identity));
  EXPECT_TRUE(swap.adjoint().isApprox(swap));
  Matrix4x4 scaled = swap;
  scaled *= 2.0;
  EXPECT_TRUE(scaled.isApprox(swap * 2.0));
  EXPECT_EQ(identity.trace(), Complex(4.0, 0.0));
  EXPECT_EQ(identity.determinant(), Complex(1.0, 0.0));
  EXPECT_EQ(swap.determinant(), Complex(-1.0, 0.0));
}

TEST(UnitaryMatrix4x4, PremultiplyBy) {
  const Matrix4x4 swap = swapMatrix();
  const Matrix4x4 phase = Matrix4x4::identity() * std::exp(1i * 0.25);
  Matrix4x4 acc = Matrix4x4::identity();
  acc.premultiplyBy(swap);
  acc.premultiplyBy(phase);
  EXPECT_TRUE(acc.isApprox(phase * swap));
}

TEST(UnitaryMatrix4x4, IsApprox) {
  const Matrix4x4 a = Matrix4x4::identity();
  Matrix4x4 b = a;
  b(3, 3) += 1e-15;
  EXPECT_TRUE(a.isApprox(b));
  EXPECT_FALSE(a.isApprox(swapMatrix()));
}

TEST(DynamicMatrix, DefaultAndSizedConstruction) {
  const DynamicMatrix empty;
  EXPECT_EQ(empty.rows(), 0);
  EXPECT_EQ(empty.cols(), 0);

  const DynamicMatrix sized(2);
  EXPECT_EQ(sized.rows(), 2);
  EXPECT_EQ(sized(0, 0), 0.0);
  EXPECT_EQ(sized(1, 1), 0.0);
}

TEST(DynamicMatrix, CopyMoveAssign) {
  DynamicMatrix original(2);
  original(0, 0) = 1.0;
  original(1, 1) = 1.0;

  DynamicMatrix copied(original);
  EXPECT_TRUE(copied.isApprox(original));

  DynamicMatrix moved(std::move(copied));
  EXPECT_TRUE(moved.isApprox(original));

  const DynamicMatrix assigned = original;
  EXPECT_TRUE(assigned.isApprox(original));

  DynamicMatrix moveAssigned(1);
  moveAssigned = std::move(moved);
  EXPECT_TRUE(moveAssigned.isApprox(original));
}

TEST(DynamicMatrix, IdentityAndElementAccess) {
  const DynamicMatrix identity = DynamicMatrix::identity(3);
  EXPECT_EQ(identity.rows(), 3);
  EXPECT_EQ(identity.cols(), 3);
  EXPECT_EQ(identity(0, 0), 1.0);
  EXPECT_EQ(identity(1, 2), 0.0);
  EXPECT_EQ(identity(2, 2), 1.0);
  EXPECT_TRUE(identity.isIdentity());

  DynamicMatrix mutableMatrix = identity;
  mutableMatrix(1, 1) = 0.5;
  EXPECT_EQ(mutableMatrix(1, 1), 0.5);
}

TEST(DynamicMatrix, FromAdjoint) {
  const Matrix2x2 x = pauliX();
  EXPECT_TRUE(DynamicMatrix::fromAdjoint(x).isApprox(x.adjoint()));
  const Complex global = std::polar(1.0, 0.25);
  EXPECT_TRUE(
      DynamicMatrix::fromAdjoint(x * global).isApprox((x * global).adjoint()));
  EXPECT_TRUE(DynamicMatrix(x).isApprox(x));
  EXPECT_TRUE(DynamicMatrix(swapMatrix()).isApprox(swapMatrix()));
}

TEST(DynamicMatrix, ScalarMultiplyAssign) {
  DynamicMatrix matrix = DynamicMatrix::identity(2);
  matrix *= std::exp(Complex{0.0, 0.5});
  EXPECT_TRUE(matrix.isApprox(DynamicMatrix::identity(2) *
                              std::exp(Complex{0.0, 0.5})));
}

TEST(DynamicMatrix, AssignFrom) {
  DynamicMatrix dynamic;

  dynamic.assignFrom(Matrix1x1::fromElements(0.25));
  EXPECT_EQ(dynamic.rows(), 1);
  EXPECT_EQ(dynamic(0, 0), 0.25);

  dynamic.assignFrom(pauliX());
  EXPECT_TRUE(dynamic.isApprox(pauliX()));

  dynamic.assignFrom(swapMatrix());
  EXPECT_TRUE(dynamic.isApprox(swapMatrix()));

  const DynamicMatrix source = DynamicMatrix::identity(2);
  dynamic.assignFrom(source);
  EXPECT_TRUE(dynamic.isApprox(source));
}

TEST(DynamicMatrix, SetBottomRightCorner) {
  const Matrix2x2 x = pauliX();
  const Matrix4x4 swap = swapMatrix();

  DynamicMatrix with2x2 = DynamicMatrix::identity(4);
  with2x2.setBottomRightCorner(x);
  EXPECT_EQ(with2x2(0, 0), 1.0);
  EXPECT_EQ(with2x2(2, 2), 0.0);
  EXPECT_EQ(with2x2(2, 3), 1.0);
  EXPECT_EQ(with2x2(3, 2), 1.0);

  DynamicMatrix with4x4 = DynamicMatrix::identity(6);
  with4x4.setBottomRightCorner(swap);
  EXPECT_EQ(with4x4(0, 0), 1.0);
  EXPECT_EQ(with4x4(1, 1), 1.0);
  EXPECT_EQ(with4x4(2, 2), 1.0);
  EXPECT_EQ(with4x4(3, 4), 1.0);
  EXPECT_EQ(with4x4(4, 3), 1.0);
  EXPECT_EQ(with4x4(5, 5), 1.0);

  DynamicMatrix block = DynamicMatrix::identity(2);
  block(0, 1) = 1i;
  DynamicMatrix withDynamic = DynamicMatrix::identity(3);
  withDynamic.setBottomRightCorner(block);
  EXPECT_EQ(withDynamic(1, 1), 1.0);
  EXPECT_EQ(withDynamic(1, 2), 1i);
  EXPECT_EQ(withDynamic(2, 1), 0.0);
}

TEST(DynamicMatrix, Adjoint) {
  DynamicMatrix matrix(2);
  matrix(0, 0) = 1.0;
  matrix(0, 1) = 1i;
  matrix(1, 0) = 0.0;
  matrix(1, 1) = 1.0;

  const DynamicMatrix adjoint = matrix.adjoint();
  EXPECT_EQ(adjoint(0, 0), 1.0);
  EXPECT_EQ(adjoint(0, 1), 0.0);
  EXPECT_EQ(adjoint(1, 0), -1i);
  EXPECT_EQ(adjoint(1, 1), 1.0);
}

TEST(DynamicMatrix, IsApproxRejectsMismatchedExtents) {
  EXPECT_FALSE(DynamicMatrix::identity(1).isApprox(DynamicMatrix::identity(2)));
}

TEST(DynamicMatrix, IsApproxOverloads) {
  const Matrix1x1 phase = Matrix1x1::fromElements(Complex{0.25, 0.5});
  const Matrix2x2 x = pauliX();
  const Matrix4x4 swap = swapMatrix();

  DynamicMatrix as1x1;
  as1x1.assignFrom(phase);
  EXPECT_TRUE(as1x1.isApprox(phase));
  EXPECT_FALSE(as1x1.isApprox(Matrix1x1::fromElements(1.0)));

  DynamicMatrix as2x2;
  as2x2.assignFrom(x);
  EXPECT_TRUE(as2x2.isApprox(x));
  EXPECT_FALSE(as2x2.isApprox(Matrix2x2::identity()));

  DynamicMatrix as4x4;
  as4x4.assignFrom(swap);
  EXPECT_TRUE(as4x4.isApprox(swap));
  EXPECT_FALSE(as4x4.isApprox(Matrix4x4::identity()));

  DynamicMatrix wrongDim = DynamicMatrix::identity(3);
  EXPECT_FALSE(wrongDim.isApprox(phase));
  EXPECT_FALSE(wrongDim.isApprox(x));
  EXPECT_FALSE(wrongDim.isApprox(swap));

  const DynamicMatrix a = DynamicMatrix::identity(2);
  DynamicMatrix b = a;
  b(1, 0) += 1e-15;
  EXPECT_TRUE(a.isApprox(b));
  EXPECT_FALSE(a.isApprox(DynamicMatrix::identity(3)));
}

TEST(Matrix1x1, AssignFromDynamicMatrix) {
  const Matrix1x1 phase = Matrix1x1::fromElements(Complex{0.25, 0.5});

  DynamicMatrix dynamic;
  dynamic.assignFrom(phase);

  Matrix1x1 out = Matrix1x1::fromElements(1.0);
  EXPECT_TRUE(out.assignFrom(dynamic));
  EXPECT_TRUE(out.isApprox(phase));
  EXPECT_FALSE(out.assignFrom(DynamicMatrix::identity(2)));
}

TEST(Matrix2x2, AssignFromDynamicMatrix) {
  const Matrix2x2 x = pauliX();

  DynamicMatrix dynamic;
  dynamic.assignFrom(x);

  Matrix2x2 out = Matrix2x2::identity();
  EXPECT_TRUE(out.assignFrom(dynamic));
  EXPECT_TRUE(out.isApprox(x));
  EXPECT_FALSE(out.assignFrom(DynamicMatrix::identity(3)));
}

TEST(Matrix4x4, AssignFromDynamicMatrix) {
  const Matrix4x4 swap = swapMatrix();

  DynamicMatrix dynamic;
  dynamic.assignFrom(swap);

  Matrix4x4 out = Matrix4x4::identity();
  EXPECT_TRUE(out.assignFrom(dynamic));
  EXPECT_TRUE(out.isApprox(swap));
  EXPECT_FALSE(out.assignFrom(DynamicMatrix::identity(2)));
}

TEST(UnitaryMatrix2x2, TransposeAndIsIdentity) {
  const Matrix2x2 m = Matrix2x2::fromElements(1, 2i, 3, 4);
  EXPECT_TRUE(m.transpose().isApprox(Matrix2x2::fromElements(1, 3, 2i, 4)));
  EXPECT_TRUE(Matrix2x2::identity().isIdentity());
  EXPECT_FALSE(pauliX().isIdentity());
  Matrix2x2 nearIdentity = Matrix2x2::identity();
  nearIdentity(0, 1) = 1e-15;
  EXPECT_TRUE(nearIdentity.isIdentity());
  nearIdentity(0, 1) = 1.0;
  EXPECT_FALSE(nearIdentity.isIdentity());
}

TEST(UnitaryMatrix4x4, TransposeAndIsIdentity) {
  Matrix4x4 m = Matrix4x4::identity();
  m(0, 3) = 2i;
  m(3, 0) = 5.0;
  const Matrix4x4 t = m.transpose();
  EXPECT_EQ(t(3, 0), 2i);
  EXPECT_EQ(t(0, 3), 5.0);
  EXPECT_TRUE(Matrix4x4::identity().isIdentity());
  EXPECT_FALSE(swapMatrix().isIdentity());
}

TEST(UnitaryMatrix4x4, DiagonalRowsColumnsAndParts) {
  Matrix4x4 m = Matrix4x4::fromElements(Complex{1, 1}, 0, 0, 0,  // row 0
                                        0, Complex{2, 2}, 0, 0,  // row 1
                                        0, 0, Complex{3, 3}, 0,  // row 2
                                        0, 0, 0, Complex{4, 4}); // row 3
  const auto diag = m.diagonal();
  EXPECT_EQ(diag[0], (Complex{1, 1}));
  EXPECT_EQ(diag[3], (Complex{4, 4}));
  EXPECT_TRUE(Matrix4x4::fromDiagonal(diag).isApprox(m));

  const auto col1 = m.column(1);
  EXPECT_EQ(col1[1], (Complex{2, 2}));
  Matrix4x4 n = Matrix4x4::identity();
  n.setColumn(2, {1i, 2i, 3i, 4i});
  EXPECT_EQ(n(0, 2), 1i);
  EXPECT_EQ(n(3, 2), 4i);

  const auto row1 = m.row(1);
  ASSERT_EQ(row1.size(), Matrix4x4::K_COLS);
  EXPECT_EQ(row1[0], (Complex{0, 0}));
  EXPECT_EQ(row1[1], (Complex{2, 2}));
  EXPECT_EQ(row1[0], m(1, 0));
  EXPECT_EQ(row1[3], m(1, 3));

  Matrix4x4 r = Matrix4x4::identity();
  r.setRow(2, {1.0, 2.0, 3.0, 4.0});
  EXPECT_EQ(r(2, 0), 1.0);
  EXPECT_EQ(r(2, 3), 4.0);
  EXPECT_EQ(r.row(2)[1], 2.0);

  const auto re = m.realPart();
  const auto im = m.imagPart();
  EXPECT_EQ(re[0], 1.0);
  EXPECT_EQ(im[0], 1.0);
  EXPECT_EQ(re[15], 4.0);
  EXPECT_EQ(im[15], 4.0);
}

TEST(UnitaryMatrix4x4, KroneckerProduct) {
  const Matrix2x2 x = pauliX();
  // X (x) I should swap the high bit.
  const Matrix4x4 xi = Matrix4x4::kron(x, Matrix2x2::identity());
  EXPECT_TRUE(xi.isApprox(Matrix4x4::fromElements(0, 0, 1, 0, // row 0
                                                  0, 0, 0, 1, // row 1
                                                  1, 0, 0, 0, // row 2
                                                  0, 1, 0, 0)));
  // I (x) X swaps the low bit.
  const Matrix4x4 ix = Matrix4x4::kron(Matrix2x2::identity(), x);
  EXPECT_TRUE(ix.isApprox(Matrix4x4::fromElements(0, 1, 0, 0, // row 0
                                                  1, 0, 0, 0, // row 1
                                                  0, 0, 0, 1, // row 2
                                                  0, 0, 1, 0)));
}

TEST(UnitaryMatrix4x4, ReorderTwoQubitMatrix) {
  const Matrix2x2 x = pauliX();
  const Matrix4x4 onHigh = Matrix4x4::kron(x, Matrix2x2::identity());
  const Matrix4x4 onLow = Matrix4x4::kron(Matrix2x2::identity(), x);

  EXPECT_TRUE(onHigh.reorderForQubits(0, 1).isApprox(onHigh));
  EXPECT_TRUE(onHigh.reorderForQubits(1, 0).isApprox(onLow));
  EXPECT_TRUE(onLow.reorderForQubits(1, 0).isApprox(onHigh));
}

TEST(UnitaryDynamicMatrix, NQubitEmbedMatchesTwoQubitSpecialization) {
  const Matrix2x2 x = pauliX();
  constexpr Matrix4x4 cx = Matrix4x4::fromElements(1, 0, 0, 0, //
                                                   0, 1, 0, 0, //
                                                   0, 0, 0, 1, //
                                                   0, 0, 1, 0);
  EXPECT_TRUE(x.embedInNqubit(2, 0).isApprox(
      Matrix4x4::kron(x, Matrix2x2::identity())));
  EXPECT_TRUE(x.embedInNqubit(2, 1).isApprox(
      Matrix4x4::kron(Matrix2x2::identity(), x)));
  EXPECT_TRUE(cx.embedInNqubit(2, 0, 1).isApprox(cx.reorderForQubits(0, 1)));
  const DynamicMatrix cxOn01 = cx.embedInNqubit(3, 0, 1);
  const DynamicMatrix cxOn12 = cx.embedInNqubit(3, 1, 2);
  EXPECT_EQ(cxOn01.rows(), 8);
  EXPECT_EQ(cxOn12.rows(), 8);
  EXPECT_FALSE(cxOn01.isApprox(cxOn12));
}

TEST(UnitaryDynamicMatrix, EmbedSingleQubitOnMiddleWire) {
  const Matrix2x2 x = pauliX();
  const DynamicMatrix embedded = x.embedInNqubit(3, 1);
  EXPECT_EQ(embedded.rows(), 8);
  EXPECT_FALSE(embedded.isIdentity());

  const DynamicMatrix product = embedded * embedded;
  EXPECT_TRUE(product.isIdentity());
  EXPECT_NEAR(product.trace().real(), 8.0, MATRIX_TOLERANCE);
}

TEST(UnitaryDynamicMatrix, MultiplyTraceAndScalar) {
  const Matrix2x2 x = pauliX();
  const DynamicMatrix embedded = x.embedInNqubit(2, 0);
  EXPECT_FALSE(embedded.isIdentity());
  const Complex scalar = std::exp(1i * 0.3);
  EXPECT_TRUE((scalar * embedded).isApprox(embedded * scalar));
  const DynamicMatrix product = embedded * embedded;
  EXPECT_TRUE(product.isIdentity());
  EXPECT_NEAR(product.trace().real(), 4.0, MATRIX_TOLERANCE);
}

TEST(DynamicMatrix, MultiplyAdjointTraceAt4) {
  const auto swap = DynamicMatrix(swapMatrix());
  EXPECT_EQ(swap.rows(), 4);

  const DynamicMatrix product = swap * swap;
  EXPECT_TRUE(product.isIdentity());
  EXPECT_NEAR(product.trace().real(), 4.0, MATRIX_TOLERANCE);

  const DynamicMatrix adjoint = swap.adjoint();
  EXPECT_TRUE(adjoint.isApprox(swapMatrix()));
}

TEST(DynamicMatrix, MultiplyAt2) {
  const DynamicMatrix x(pauliX());
  EXPECT_EQ(x.rows(), 2);
  const DynamicMatrix product = x * x;
  EXPECT_TRUE(product.isIdentity());
  EXPECT_NEAR(product.trace().real(), 2.0, MATRIX_TOLERANCE);
  EXPECT_TRUE(product.isApprox(pauliX() * pauliX()));
}

TEST(DynamicMatrix, IsIdentityOffDiagonal) {
  DynamicMatrix matrix = DynamicMatrix::identity(2);
  matrix(0, 1) = 1.0;
  EXPECT_FALSE(matrix.isIdentity());
}

TEST(UnitaryMatrix2x2, ScalarLeftMultiply) {
  const Matrix2x2 x = pauliX();
  const Complex scalar = std::exp(1i * 0.5);
  EXPECT_TRUE((scalar * x).isApprox(x * scalar));
}

TEST(UnitaryMatrix4x4, ScalarLeftMultiply) {
  const Matrix4x4 swap = swapMatrix();
  const Complex scalar = std::exp(1i * 0.25);
  EXPECT_TRUE((scalar * swap).isApprox(swap * scalar));
}

TEST(SymmetricEigensolver, DiagonalMatrix) {
  std::array<double, 16> a{};
  a[0] = 3.0;
  a[5] = 1.0;
  a[10] = 4.0;
  a[15] = 2.0;
  const SymmetricEigenDecomposition4x4 result =
      Matrix4x4::fromRealRowMajor(a).symmetricEigenDecomposition();
  EXPECT_NEAR(result.eigenvalues[0], 1.0, MATRIX_TOLERANCE);
  EXPECT_NEAR(result.eigenvalues[1], 2.0, MATRIX_TOLERANCE);
  EXPECT_NEAR(result.eigenvalues[2], 3.0, MATRIX_TOLERANCE);
  EXPECT_NEAR(result.eigenvalues[3], 4.0, MATRIX_TOLERANCE);
}

TEST(SymmetricEigensolver, Matrix4x4Overload) {
  std::array<double, 16> a{};
  a[0] = 3.0;
  a[5] = 1.0;
  a[10] = 4.0;
  a[15] = 2.0;
  const Matrix4x4 matrix = Matrix4x4::fromElements(3.0, 0.0, 0.0, 0.0,  // row 0
                                                   0.0, 1.0, 0.0, 0.0,  // row 1
                                                   0.0, 0.0, 4.0, 0.0,  // row 2
                                                   0.0, 0.0, 0.0, 2.0); // row 3
  const SymmetricEigenDecomposition4x4 fromArray =
      Matrix4x4::fromRealRowMajor(a).symmetricEigenDecomposition();
  const SymmetricEigenDecomposition4x4 fromMatrix =
      matrix.symmetricEigenDecomposition();
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(fromMatrix.eigenvalues[i], fromArray.eigenvalues[i],
                MATRIX_TOLERANCE);
  }
  EXPECT_TRUE(fromMatrix.eigenvectors.isApprox(fromArray.eigenvectors));
}

TEST(SymmetricEigensolver, ReconstructsRandomSymmetric) {
  std::mt19937 rng(0xC0FFEE);
  std::uniform_real_distribution dist(-2.0, 2.0);
  for (int trial = 0; trial < 50; ++trial) {
    std::array<double, 16> a{};
    for (std::size_t i = 0; i < 4; ++i) {
      for (std::size_t j = i; j < 4; ++j) {
        const double value = dist(rng);
        a[(i * 4) + j] = value;
        a[(j * 4) + i] = value;
      }
    }
    const SymmetricEigenDecomposition4x4 result =
        Matrix4x4::fromRealRowMajor(a).symmetricEigenDecomposition();

    // Eigenvalues are ascending.
    for (std::size_t i = 0; i + 1 < 4; ++i) {
      EXPECT_LE(result.eigenvalues[i],
                result.eigenvalues[i + 1] + MATRIX_TOLERANCE);
    }

    // Eigenvectors are orthonormal: V^T V == I.
    const Matrix4x4& v = result.eigenvectors;
    EXPECT_TRUE((v.transpose() * v).isIdentity());

    // Reconstruction: V D V^T == A.
    const Matrix4x4 d =
        Matrix4x4::fromDiagonal({result.eigenvalues[0], result.eigenvalues[1],
                                 result.eigenvalues[2], result.eigenvalues[3]});
    const Matrix4x4 reconstructed = v * d * v.transpose();
    const Matrix4x4 original =
        Matrix4x4::fromElements(a[0], a[1], a[2], a[3],      // row 0
                                a[4], a[5], a[6], a[7],      // row 1
                                a[8], a[9], a[10], a[11],    // row 2
                                a[12], a[13], a[14], a[15]); // row 3
    EXPECT_TRUE(reconstructed.isApprox(original));
  }
}

TEST(SymmetricEigensolver, HandlesDegenerateSpectrum) {
  // A scalar multiple of the identity: every vector is an eigenvector, but the
  // returned basis must still be orthonormal.
  std::array<double, 16> a{};
  for (std::size_t i = 0; i < 4; ++i) {
    a[(i * 4) + i] = 2.5;
  }
  const SymmetricEigenDecomposition4x4 result =
      Matrix4x4::fromRealRowMajor(a).symmetricEigenDecomposition();
  for (const double value : result.eigenvalues) {
    EXPECT_NEAR(value, 2.5, MATRIX_TOLERANCE);
  }
  const Matrix4x4& v = result.eigenvectors;
  EXPECT_TRUE((v.transpose() * v).isIdentity());
}

TEST(GlobalPhaseEquivalence, IsEquivalentUpToGlobalPhase) {
  const Matrix2x2 pauliX = Matrix2x2::fromElements(0, 1, 1, 0);
  const Complex phase{0.0, 1.0};
  EXPECT_TRUE(isEquivalentUpToGlobalPhase(pauliX, phase * pauliX));
  EXPECT_TRUE(isEquivalentUpToGlobalPhase(pauliX, pauliX));
  EXPECT_FALSE(isEquivalentUpToGlobalPhase(pauliX, Matrix2x2::identity()));
}

TEST(Eigensolver, EmptyMatrixReturnsNullopt) {
  EXPECT_FALSE(DynamicMatrix{}.eigenDecomposition().has_value());
}

TEST(Eigensolver, Scalar1x1) {
  DynamicMatrix matrix(1);
  matrix(0, 0) = std::exp(1i * 0.7);
  expectEigenDecomposition(matrix);
}

TEST(Eigensolver, DiagonalReal4x4GeneralEigenpairs) {
  expectGeneralEigenDecomposition(
      DynamicMatrix(Matrix4x4::fromDiagonal({1.0, 2.0, 3.0, 4.0})));
}

TEST(Eigensolver, SwapMatrixGeneralEigenpairs) {
  expectGeneralEigenDecomposition(DynamicMatrix(swapMatrix()));
}

TEST(Eigensolver, GeneralComplex2x2ClosedForm) {
  expectGeneralEigenDecomposition(DynamicMatrix(
      Matrix2x2::fromElements(Complex{1.0, 1.0}, Complex{2.0, -1.0}, // row 0
                              Complex{3.0, 0.5}, Complex{4.0, -2.0}  // row 1
                              )));
}

TEST(Eigensolver, GeneralComplex4x4EispackEigenvalues) {
  const Matrix4x4 matrix = nonDiagonalTestMatrix4x4();

  const std::optional<EigenDecomposition4x4> decomposition =
      matrix.eigenDecomposition();
  ASSERT_TRUE(decomposition.has_value());

  const std::array<Complex, 4> expectedEigenvalues = {
      Complex{-1.96969058, 0.0},
      Complex{1.62487551, 0.974262379},
      Complex{1.62487551, -0.974262379},
      Complex{1.21993957, 0.0},
  };
  for (const Complex& expected : expectedEigenvalues) {
    const bool matched = std::ranges::any_of(
        decomposition->eigenvalues, [&](const Complex& actual) {
          return complexesAreApprox(actual, expected, 1e-6);
        });
    EXPECT_TRUE(matched) << "missing eigenvalue " << expected;
  }
}

TEST(Eigensolver, GeneralComplex4x4EispackEigenpairs) {
  expectGeneralEigenDecomposition(DynamicMatrix(nonDiagonalTestMatrix4x4()));
}

TEST(Eigensolver, Matrix2x2DiagonalClosedForm) {
  expectGeneralEigenDecomposition(
      DynamicMatrix(Matrix2x2::fromElements(1.0, 0.0, // row 0
                                            0.0, 2.0  // row 1
                                            )));
}

TEST(Eigensolver, Matrix2x2IdentityFullBasis) {
  const std::optional<EigenDecomposition2x2> eigen =
      Matrix2x2::identity().eigenDecomposition();
  ASSERT_TRUE(eigen.has_value());
  EXPECT_TRUE(
      eigen->eigenvectors.isApprox(Matrix2x2::identity(), MATRIX_TOLERANCE));
  EXPECT_NEAR(std::abs(eigen->eigenvectors.determinant()), 1.0,
              MATRIX_TOLERANCE);
}

TEST(Eigensolver, Matrix2x2LowerTriangularClosedForm) {
  const Matrix2x2 matrix = Matrix2x2::fromElements(1.0, 0.0, // row 0
                                                   2.0, 3.0  // row 1
  );
  expectGeneralEigenDecomposition(DynamicMatrix(matrix));
}

TEST(Eigensolver, Matrix2x2PauliXDirect) {
  const std::optional<EigenDecomposition2x2> direct =
      pauliX().eigenDecomposition();
  ASSERT_TRUE(direct.has_value());
  expectGeneralEigenDecomposition(DynamicMatrix(pauliX()));
}

TEST(Eigensolver, Matrix4x4SwapDirect) {
  const std::optional<EigenDecomposition4x4> direct =
      swapMatrix().eigenDecomposition();
  ASSERT_TRUE(direct.has_value());
  expectGeneralEigenDecomposition(DynamicMatrix(swapMatrix()));
}

TEST(Eigensolver, GeneralComplex3x3DynamicPath) {
  std::mt19937 rng(0x33U); // NOLINT(cert-msc51-cpp)
  expectGeneralEigenDecomposition(randomComplexMatrix(3, rng));
}

TEST(Eigensolver, GeneralComplex8x8DynamicPath) {
  std::mt19937 rng(0x88U); // NOLINT(cert-msc51-cpp)
  expectGeneralEigenDecomposition(randomComplexMatrix(8, rng));
}

TEST(Eigensolver, RandomComplex4x4Eispack) {
  std::mt19937 rng(0x44U); // NOLINT(cert-msc51-cpp)
  for (int trial = 0; trial < 20; ++trial) {
    expectGeneralEigenDecomposition(randomComplexMatrix(4, rng));
  }
}

TEST(Eigensolver, DirectDecomposeHelpers) {
  const Matrix4x4 diagonal = Matrix4x4::fromDiagonal({1.0, 2.0, 3.0, 4.0});
  const SymmetricEigenDecomposition4x4 symmetric =
      diagonal.symmetricEigenDecomposition();
  EXPECT_NEAR(symmetric.eigenvalues[0], 1.0, MATRIX_TOLERANCE);
  EXPECT_NEAR(symmetric.eigenvalues[3], 4.0, MATRIX_TOLERANCE);

  DynamicMatrix scalar(1);
  scalar(0, 0) = Complex{2.0, -1.0};
  const std::optional<EigenDecomposition> oneByOne =
      scalar.eigenDecomposition();
  ASSERT_TRUE(oneByOne.has_value());
  EXPECT_TRUE(complexesAreApprox(oneByOne->eigenvalues[0], scalar(0, 0),
                                 MATRIX_TOLERANCE));

  const Matrix2x2 twoByTwo = pauliX();
  const std::optional<EigenDecomposition2x2> eigen2 =
      twoByTwo.eigenDecomposition();
  ASSERT_TRUE(eigen2.has_value());
  const EigenDecomposition lifted2 = EigenDecomposition::from(*eigen2);
  expectGeneralEigenDecomposition(DynamicMatrix(twoByTwo));
  EXPECT_TRUE(
      lifted2.eigenvectors.isApprox(DynamicMatrix(eigen2->eigenvectors)));

  const std::optional<EigenDecomposition4x4> eigen4 =
      nonDiagonalTestMatrix4x4().eigenDecomposition();
  ASSERT_TRUE(eigen4.has_value());
  const EigenDecomposition lifted4 = EigenDecomposition::from(*eigen4);
  EXPECT_TRUE(
      lifted4.eigenvectors.isApprox(DynamicMatrix(eigen4->eigenvectors)));
}

TEST(SymmetricEigensolver, SparseCornerElement) {
  std::array<double, 16> sparse{};
  sparse[15] = 7.0;
  const SymmetricEigenDecomposition4x4 fromArray =
      Matrix4x4::fromRealRowMajor(sparse).symmetricEigenDecomposition();
  EXPECT_NEAR(fromArray.eigenvalues[3], 7.0, MATRIX_TOLERANCE);
  for (std::size_t i = 0; i + 1 < 4; ++i) {
    EXPECT_NEAR(fromArray.eigenvalues[i], 0.0, MATRIX_TOLERANCE);
  }
}

TEST(Eigensolver, RandomComplex2x2ClosedForm) {
  std::mt19937 rng(0xE1E1E1U); // NOLINT(cert-msc51-cpp)
  for (int trial = 0; trial < 50; ++trial) {
    expectGeneralEigenDecomposition(randomComplexMatrix(2, rng));
  }
}

TEST_P(EigenDecompositionFixedDynamicTest, SpecializedPathMatchesDynamic) {
  GetParam().verify();
}

TEST_P(EigenDecompositionMatrixTest, UnitaryMatrix) {
  expectEigenDecomposition(GetParam().matrix);
}

INSTANTIATE_TEST_SUITE_P(
    Eigensolver, EigenDecompositionFixedDynamicTest,
    testing::Values(
        FixedDynamicEigenCase{"Matrix2x2", verifyMatrix2x2FixedMatchesDynamic},
        FixedDynamicEigenCase{"Matrix4x4", verifyMatrix4x4FixedMatchesDynamic}),
    [](const testing::TestParamInfo<FixedDynamicEigenCase>& info) {
      return info.param.name;
    });

INSTANTIATE_TEST_SUITE_P(
    Diagonal, EigenDecompositionMatrixTest,
    testing::Values(EigenDecompositionMatrixCase{"2x2", diagonalUnitary(2)},
                    EigenDecompositionMatrixCase{"4x4", diagonalUnitary(4)},
                    EigenDecompositionMatrixCase{"8x8", diagonalUnitary(8)}),
    [](const testing::TestParamInfo<EigenDecompositionMatrixCase>& info) {
      return info.param.name;
    });

INSTANTIATE_TEST_SUITE_P(
    NonDiagonal, EigenDecompositionMatrixTest,
    testing::Values(
        EigenDecompositionMatrixCase{"4x4Swap", DynamicMatrix(swapMatrix())},
        EigenDecompositionMatrixCase{"8x8Cyclic", cyclicPermutation(8)}),
    [](const testing::TestParamInfo<EigenDecompositionMatrixCase>& info) {
      return info.param.name;
    });
