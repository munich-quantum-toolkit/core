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

#include <Eigen/Core>
#include <cmath>
#include <complex>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Region.h>
#include <numbers>

namespace mlir::utils {

inline Eigen::Matrix2cd getMatrixId() {
  using namespace std::complex_literals;
  return Eigen::Matrix2cd{{1.0 + 0i, 0.0 + 0i},  // row 0
                          {0.0 + 0i, 1.0 + 0i}}; // row 1
}

inline Eigen::Matrix2cd getMatrixX() {
  using namespace std::complex_literals;
  return Eigen::Matrix2cd{{0.0 + 0i, 1.0 + 0i},  // row 0
                          {1.0 + 0i, 0.0 + 0i}}; // row 1
}

inline Eigen::Matrix2cd getMatrixY() {
  using namespace std::complex_literals;
  return Eigen::Matrix2cd{{0.0 + 0i, 0.0 - 1i},  // row 0
                          {0.0 + 1i, 0.0 + 0i}}; // row 1
}

inline Eigen::Matrix2cd getMatrixZ() {
  using namespace std::complex_literals;
  return Eigen::Matrix2cd{{1.0 + 0i, 0.0 + 0i},   // row 0
                          {0.0 + 0i, -1.0 + 0i}}; // row 1
}

inline Eigen::Matrix2cd getMatrixH() {
  using namespace std::complex_literals;
  const std::complex<double> m00 = 1.0 / std::sqrt(2) + 0i;
  const std::complex<double> m11 = -1.0 / std::sqrt(2) + 0i;
  return Eigen::Matrix2cd{{m00, m00}, {m00, m11}};
}

inline Eigen::Matrix2cd getMatrixS() {
  using namespace std::complex_literals;
  return Eigen::Matrix2cd{{1.0 + 0i, 0.0 + 0i},  // row 0
                          {0.0 + 0i, 0.0 + 1i}}; // row 1
}

inline Eigen::Matrix2cd getMatrixSdg() {
  using namespace std::complex_literals;
  return Eigen::Matrix2cd{{1.0 + 0i, 0.0 + 0i},  // row 0
                          {0.0 + 0i, 0.0 - 1i}}; // row 1
}

inline Eigen::Matrix2cd getMatrixT() {
  using namespace std::complex_literals;
  const std::complex<double> m00 = 1.0 + 0i;
  const std::complex<double> m01 = 0.0 + 0i;
  const std::complex<double> m11 = std::exp(1i * std::numbers::pi / 4.0);
  return Eigen::Matrix2cd{{m00, m01}, {m01, m11}};
}

inline Eigen::Matrix2cd getMatrixTdg() {
  using namespace std::complex_literals;
  const std::complex<double> m00 = 1.0 + 0i;
  const std::complex<double> m01 = 0.0 + 0i;
  const std::complex<double> m11 = std::exp(-1i * std::numbers::pi / 4.0);
  return Eigen::Matrix2cd{{m00, m01}, {m01, m11}};
}

inline Eigen::Matrix2cd getMatrixSX() {
  using namespace std::complex_literals;
  const std::complex<double> m00 = (1.0 + 1i) / 2.0;
  const std::complex<double> m01 = (1.0 - 1i) / 2.0;
  return Eigen::Matrix2cd{{m00, m01}, {m01, m00}};
}

inline Eigen::Matrix2cd getMatrixSXdg() {
  using namespace std::complex_literals;
  const std::complex<double> m00 = (1.0 - 1i) / 2.0;
  const std::complex<double> m01 = (1.0 + 1i) / 2.0;
  return Eigen::Matrix2cd{{m00, m01}, {m01, m00}};
}

inline Eigen::Matrix2cd getMatrixRX(double theta) {
  using namespace std::complex_literals;
  const std::complex<double> m00 = std::cos(theta / 2.0) + 0i;
  const std::complex<double> m01 = -1i * std::sin(theta / 2.0);
  return Eigen::Matrix2cd{{m00, m01}, {m01, m00}};
}

inline Eigen::Matrix2cd getMatrixRY(double theta) {
  using namespace std::complex_literals;
  const std::complex<double> m00 = std::cos(theta / 2.0) + 0i;
  const std::complex<double> m01 = -std::sin(theta / 2.0) + 0i;
  return Eigen::Matrix2cd{{m00, m01}, {-m01, m00}};
}

inline Eigen::Matrix2cd getMatrixRZ(double theta) {
  using namespace std::complex_literals;
  const std::complex<double> m00 = std::exp(-1i * theta / 2.0);
  const std::complex<double> m01 = 0.0 + 0i;
  const std::complex<double> m11 = std::exp(1i * theta / 2.0);
  return Eigen::Matrix2cd{{m00, m01}, {m01, m11}};
}

inline Eigen::Matrix2cd getMatrixP(double theta) {
  using namespace std::complex_literals;
  const std::complex<double> m00 = 1.0 + 0i;
  const std::complex<double> m01 = 0.0 + 0i;
  const std::complex<double> m11 = std::exp(1i * theta);
  return Eigen::Matrix2cd{{m00, m01}, {m01, m11}};
}

inline Eigen::Matrix2cd getMatrixU2(double phi, double lambda) {
  using namespace std::complex_literals;
  const std::complex<double> m00 = 1.0 / std::sqrt(2) + 0i;
  const std::complex<double> m01 = -std::exp(1i * lambda) / std::sqrt(2);
  const std::complex<double> m10 = std::exp(1i * phi) / std::sqrt(2);
  const std::complex<double> m11 = std::exp(1i * (phi + lambda)) / std::sqrt(2);
  return Eigen::Matrix2cd{{m00, m01}, {m10, m11}};
}

inline Eigen::Matrix2cd getMatrixU(double theta, double phi, double lambda) {
  using namespace std::complex_literals;
  const std::complex<double> m00 = std::cos(theta / 2.0) + 0i;
  const std::complex<double> m01 =
      -std::exp(1i * lambda) * std::sin(theta / 2.0);
  const std::complex<double> m10 = std::exp(1i * phi) * std::sin(theta / 2.0);
  const std::complex<double> m11 =
      std::exp(1i * (phi + lambda)) * std::cos(theta / 2.0);
  return Eigen::Matrix2cd{{m00, m01}, {m10, m11}};
}

inline Eigen::Matrix2cd getMatrixR(double theta, double phi) {
  using namespace std::complex_literals;
  const std::complex<double> m00 = std::cos(theta / 2.0) + 0i;
  const std::complex<double> m01 =
      -1i * std::exp(-1i * phi) * std::sin(theta / 2.0);
  const std::complex<double> m10 =
      -1i * std::exp(1i * phi) * std::sin(theta / 2.0);
  const std::complex<double> m11 = std::cos(theta / 2.0) + 0i;
  return Eigen::Matrix2cd{{m00, m01}, {m10, m11}};
}

inline Eigen::Matrix4cd getMatrixSWAP() {
  using namespace std::complex_literals;
  return Eigen::Matrix4cd{{1.0 + 0i, 0.0 + 0i, 0.0 + 0i, 0.0 + 0i},  // row 0
                          {0.0 + 0i, 0.0 + 0i, 1.0 + 0i, 0.0 + 0i},  // row 1
                          {0.0 + 0i, 1.0 + 0i, 0.0 + 0i, 0.0 + 0i},  // row 2
                          {0.0 + 0i, 0.0 + 0i, 0.0 + 0i, 1.0 + 0i}}; // row 3
}

inline Eigen::Matrix4cd getMatrixiSWAP() {
  using namespace std::complex_literals;
  return Eigen::Matrix4cd{{1.0 + 0i, 0.0 + 0i, 0.0 + 0i, 0.0 + 0i},  // row 0
                          {0.0 + 0i, 0.0 + 0i, 0.0 + 1i, 0.0 + 0i},  // row 1
                          {0.0 + 0i, 0.0 + 1i, 0.0 + 0i, 0.0 + 0i},  // row 2
                          {0.0 + 0i, 0.0 + 0i, 0.0 + 0i, 1.0 + 0i}}; // row 3
}

inline Eigen::Matrix4cd getMatrixDCX() {
  using namespace std::complex_literals;
  return Eigen::Matrix4cd{{1.0 + 0i, 0.0 + 0i, 0.0 + 0i, 0.0 + 0i},  // row 0
                          {0.0 + 0i, 0.0 + 0i, 1.0 + 0i, 0.0 + 0i},  // row 1
                          {0.0 + 0i, 0.0 + 0i, 0.0 + 0i, 1.0 + 0i},  // row 2
                          {0.0 + 0i, 1.0 + 0i, 0.0 + 0i, 0.0 + 0i}}; // row 3
}

inline Eigen::Matrix4cd getMatrixECR() {
  using namespace std::complex_literals;
  const std::complex<double> m0 = 0.0 + 0i;
  const std::complex<double> m1 = 1.0 / std::sqrt(2) + 0i;
  const std::complex<double> mi = 0.0 + 1i / std::sqrt(2);
  return Eigen::Matrix4cd{{m0, m0, m1, mi},   // row 0
                          {m0, m0, mi, m1},   // row 1
                          {m1, -mi, m0, m0},  // row 2
                          {-mi, m1, m0, m0}}; // row 3
}

inline Eigen::Matrix4cd getMatrixRXX(double theta) {
  using namespace std::complex_literals;
  const std::complex<double> m0 = 0.0 + 0i;
  const std::complex<double> mc = std::cos(theta / 2.0) + 0i;
  const std::complex<double> ms = -1i * std::sin(theta / 2.0);
  return Eigen::Matrix4cd{{mc, m0, m0, ms},  // row 0
                          {m0, mc, ms, m0},  // row 1
                          {m0, ms, mc, m0},  // row 2
                          {ms, m0, m0, mc}}; // row 3
}

inline Eigen::Matrix4cd getMatrixRYY(double theta) {
  using namespace std::complex_literals;
  const std::complex<double> m0 = 0.0 + 0i;
  const std::complex<double> mc = std::cos(theta / 2.0) + 0i;
  const std::complex<double> ms = 1i * std::sin(theta / 2.0);
  return Eigen::Matrix4cd{{mc, m0, m0, ms},  // row 0
                          {m0, mc, -ms, m0}, // row 1
                          {m0, -ms, mc, m0}, // row 2
                          {ms, m0, m0, mc}}; // row 3
}

inline Eigen::Matrix4cd getMatrixRZX(double theta) {
  using namespace std::complex_literals;
  const std::complex<double> m0 = 0.0 + 0i;
  const std::complex<double> mc = std::cos(theta / 2.0) + 0i;
  const std::complex<double> ms = -1i * std::sin(theta / 2.0);
  return Eigen::Matrix4cd{{mc, -ms, m0, m0}, // row 0
                          {-ms, mc, m0, m0}, // row 1
                          {m0, m0, mc, ms},  // row 2
                          {m0, m0, ms, mc}}; // row 3
}

inline Eigen::Matrix4cd getMatrixRZZ(double theta) {
  using namespace std::complex_literals;
  const std::complex<double> m0 = 0.0 + 0i;
  const std::complex<double> mp = std::exp(1i * theta / 2.0);
  const std::complex<double> mm = std::exp(-1i * theta / 2.0);
  return Eigen::Matrix4cd{{mm, m0, m0, m0},  // row 0
                          {m0, mp, m0, m0},  // row 1
                          {m0, m0, mp, m0},  // row 2
                          {m0, m0, m0, mm}}; // row 3
}

inline Eigen::Matrix4cd getMatrixXXPlusYY(double theta, double beta) {
  using namespace std::complex_literals;
  const std::complex<double> m0 = 0.0 + 0i;
  const std::complex<double> m1 = 1.0 + 0i;
  const std::complex<double> mc = std::cos(theta / 2.0) + 0i;
  const std::complex<double> msp =
      -1i * std::sin(theta / 2.0) * std::exp(1i * beta);
  const std::complex<double> msm =
      -1i * std::sin(theta / 2.0) * std::exp(-1i * beta);
  return Eigen::Matrix4cd{{m1, m0, m0, m0},  // row 0
                          {m0, mc, msm, m0}, // row 1
                          {m0, msp, mc, m0}, // row 2
                          {m0, m0, m0, m1}}; // row 3
}

inline Eigen::Matrix4cd getMatrixXXMinusYY(double theta, double beta) {
  using namespace std::complex_literals;
  const std::complex<double> m0 = 0.0 + 0i;
  const std::complex<double> m1 = 1.0 + 0i;
  const std::complex<double> mc = std::cos(theta / 2.0) + 0i;
  const std::complex<double> msp =
      std::sin(theta / 2.0) * std::exp(-1i * beta) + 0i;
  const std::complex<double> msm =
      -std::sin(theta / 2.0) * std::exp(1i * beta) + 0i;
  return Eigen::Matrix4cd{{mc, m0, m0, msm},  // row 0
                          {m0, m1, m0, m0},   // row 1
                          {m0, m0, m1, m0},   // row 2
                          {msp, m0, m0, mc}}; // row 3
}

inline Eigen::MatrixXcd getMatrixCtrl(size_t numControls,
                                      Eigen::MatrixXcd targetMatrix) {
  // get dimensions of target matrix
  const auto targetDim = targetMatrix.cols();
  assert(targetMatrix.cols() == targetMatrix.rows());

  // define dimensions and type of output matrix
  const auto dim = static_cast<int64_t>((1 << numControls) * targetDim);

  // initialize result with identity
  Eigen::MatrixXcd matrix = Eigen::MatrixXcd::Identity(dim, dim);

  // Apply target matrix
  matrix.bottomRightCorner(targetDim, targetDim) = targetMatrix;

  return matrix;
}

} // namespace mlir::utils
