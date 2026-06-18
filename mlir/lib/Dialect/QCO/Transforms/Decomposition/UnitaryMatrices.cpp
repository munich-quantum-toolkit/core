/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/UnitaryMatrices.h"

#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>

#include <cassert>
#include <cmath>
#include <complex>

namespace mlir::qco::decomposition {

Matrix2x2 uMatrix(double theta, double phi, double lambda) {
  const auto cosHalf = std::cos(theta / 2.);
  const auto sinHalf = std::sin(theta / 2.);
  return Matrix2x2::fromElements(
      Complex{cosHalf, 0.},
      Complex{-std::cos(lambda) * sinHalf, -std::sin(lambda) * sinHalf},
      Complex{std::cos(phi) * sinHalf, std::sin(phi) * sinHalf},
      Complex{std::cos(lambda + phi) * cosHalf,
              std::sin(lambda + phi) * cosHalf});
}

Matrix2x2 u2Matrix(double phi, double lambda) {
  return Matrix2x2::fromElements(
      Complex{FRAC1_SQRT2, 0.},
      Complex{-std::cos(lambda) * FRAC1_SQRT2, -std::sin(lambda) * FRAC1_SQRT2},
      Complex{std::cos(phi) * FRAC1_SQRT2, std::sin(phi) * FRAC1_SQRT2},
      Complex{std::cos(lambda + phi) * FRAC1_SQRT2,
              std::sin(lambda + phi) * FRAC1_SQRT2});
}

Matrix2x2 rxMatrix(double theta) {
  const auto halfTheta = theta / 2.;
  const Complex cos{std::cos(halfTheta), 0.};
  const Complex isin{0., -std::sin(halfTheta)};
  return Matrix2x2::fromElements(cos, isin, isin, cos);
}

Matrix2x2 ryMatrix(double theta) {
  const auto halfTheta = theta / 2.;
  const Complex cos{std::cos(halfTheta), 0.};
  const Complex sin{std::sin(halfTheta), 0.};
  return Matrix2x2::fromElements(cos, -sin, sin, cos);
}

Matrix2x2 rzMatrix(double theta) {
  return Matrix2x2::fromElements(
      Complex{std::cos(theta / 2.), -std::sin(theta / 2.)}, 0., 0.,
      Complex{std::cos(theta / 2.), std::sin(theta / 2.)});
}

Matrix4x4 rxxMatrix(double theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const Complex misin{0., -std::sin(theta / 2.)};
  return Matrix4x4::fromElements(cosTheta, 0, 0, misin, //
                                 0, cosTheta, misin, 0, //
                                 0, misin, cosTheta, 0, //
                                 misin, 0, 0, cosTheta);
}

Matrix4x4 ryyMatrix(double theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const Complex isin{0., std::sin(theta / 2.)};
  const Complex misin{0., -std::sin(theta / 2.)};
  return Matrix4x4::fromElements(cosTheta, 0, 0, isin,  //
                                 0, cosTheta, misin, 0, //
                                 0, misin, cosTheta, 0, //
                                 isin, 0, 0, cosTheta);
}

Matrix4x4 rzzMatrix(double theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);
  const Complex em{cosTheta, -sinTheta};
  const Complex ep{cosTheta, sinTheta};
  return Matrix4x4::fromElements(em, 0, 0, 0, //
                                 0, ep, 0, 0, //
                                 0, 0, ep, 0, //
                                 0, 0, 0, em);
}

Matrix2x2 pMatrix(double lambda) {
  return Matrix2x2::fromElements(1., 0., 0.,
                                 Complex{std::cos(lambda), std::sin(lambda)});
}

const Matrix4x4& swapGate() {
  static const Matrix4x4 matrix = Matrix4x4::fromElements(1, 0, 0, 0, //
                                                          0, 0, 1, 0, //
                                                          0, 1, 0, 0, //
                                                          0, 0, 0, 1);
  return matrix;
}

const Matrix2x2& hGate() {
  static const Matrix2x2 matrix = Matrix2x2::fromElements(
      FRAC1_SQRT2, FRAC1_SQRT2, FRAC1_SQRT2, -FRAC1_SQRT2);
  return matrix;
}

const Matrix2x2& ipz() {
  static const Matrix2x2 matrix =
      Matrix2x2::fromElements(Complex{0, 1}, 0, 0, Complex{0, -1});
  return matrix;
}

const Matrix2x2& ipy() {
  static const Matrix2x2 matrix = Matrix2x2::fromElements(0, 1, -1, 0);
  return matrix;
}

const Matrix2x2& ipx() {
  static const Matrix2x2 matrix =
      Matrix2x2::fromElements(0, Complex{0, 1}, Complex{0, 1}, 0);
  return matrix;
}

const Matrix4x4& cxGate01() {
  static const Matrix4x4 matrix = Matrix4x4::fromElements(1, 0, 0, 0, //
                                                          0, 1, 0, 0, //
                                                          0, 0, 0, 1, //
                                                          0, 0, 1, 0);
  return matrix;
}

const Matrix4x4& cxGate10() {
  static const Matrix4x4 matrix = Matrix4x4::fromElements(1, 0, 0, 0, //
                                                          0, 0, 0, 1, //
                                                          0, 0, 1, 0, //
                                                          0, 1, 0, 0);
  return matrix;
}

const Matrix4x4& czGate() {
  static const Matrix4x4 matrix = Matrix4x4::fromElements(1, 0, 0, 0, //
                                                          0, 1, 0, 0, //
                                                          0, 0, 1, 0, //
                                                          0, 0, 0, -1);
  return matrix;
}

Matrix4x4 expandToTwoQubits(const Matrix2x2& singleQubitMatrix,
                            QubitId qubitId) {
  if (qubitId == 0) {
    return kron(singleQubitMatrix, Matrix2x2::identity());
  }
  if (qubitId == 1) {
    return kron(Matrix2x2::identity(), singleQubitMatrix);
  }
  llvm::reportFatalInternalError("Invalid qubit id for single-qubit expansion");
}

Matrix4x4
fixTwoQubitMatrixQubitOrder(const Matrix4x4& twoQubitMatrix,
                            const llvm::SmallVector<QubitId, 2>& qubitIds) {
  if (qubitIds == llvm::SmallVector<QubitId, 2>{1, 0}) {
    // `UnitaryOpInterface::getUnitaryMatrix4x4` uses a fixed index order;
    // conjugate by SWAP when operand order is (1, 0) instead of (0, 1).
    return swapGate() * twoQubitMatrix * swapGate();
  }
  if (qubitIds == llvm::SmallVector<QubitId, 2>{0, 1}) {
    return twoQubitMatrix;
  }
  llvm::reportFatalInternalError(
      "Invalid qubit IDs for fixing two-qubit matrix");
}

} // namespace mlir::qco::decomposition
