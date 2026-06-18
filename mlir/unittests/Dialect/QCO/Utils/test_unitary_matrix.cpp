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

#include <cmath>
#include <complex>
#include <utility>

using namespace mlir::qco;
using namespace std::complex_literals;

static_assert(is_supported_matrix_v<Matrix1x1>);
static_assert(is_supported_matrix_v<Matrix2x2>);
static_assert(is_supported_matrix_v<Matrix4x4>);
static_assert(is_supported_matrix_v<DynamicMatrix>);
static_assert(!is_supported_matrix_v<int>);

[[nodiscard]] static Matrix2x2 pauliX() {
  return Matrix2x2::fromElements(0, 1, 1, 0);
}

[[nodiscard]] static Matrix4x4 swapMatrix() {
  return Matrix4x4::fromElements(1, 0, 0, 0,  // row 0
                                 0, 0, 1, 0,  // row 1
                                 0, 1, 0, 0,  // row 2
                                 0, 0, 0, 1); // row 3
}

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
  EXPECT_TRUE(identity.isApprox(
      Matrix4x4::fromElements(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)));
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
