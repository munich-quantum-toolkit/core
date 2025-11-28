/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <cmath>
#include <complex>
#include <numbers>

namespace mlir::utils {

using namespace std::complex_literals;

inline DenseElementsAttr getMatrixId(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const auto& matrix = {1.0 + 0i, 0.0 + 0i,  // row 0
                        0.0 + 0i, 1.0 + 0i}; // row 1
  return DenseElementsAttr::get(type, matrix);
}

inline DenseElementsAttr getMatrixX(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const auto& matrix = {0.0 + 0i, 1.0 + 0i,  // row 0
                        1.0 + 0i, 0.0 + 0i}; // row 1
  return DenseElementsAttr::get(type, matrix);
}

inline DenseElementsAttr getMatrixY(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const auto& matrix = {0.0 + 0i, 0.0 - 1i,  // row 0
                        0.0 + 1i, 0.0 + 0i}; // row 1
  return DenseElementsAttr::get(type, matrix);
}

inline DenseElementsAttr getMatrixZ(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const auto& matrix = {1.0 + 0i, 0.0 + 0i,   // row 0
                        0.0 + 0i, -1.0 + 0i}; // row 1
  return DenseElementsAttr::get(type, matrix);
}

inline DenseElementsAttr getMatrixH(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const std::complex<double> m00 = 1.0 / std::sqrt(2) + 0i;
  const std::complex<double> m11 = -1.0 / std::sqrt(2) + 0i;
  return DenseElementsAttr::get(type, {m00, m00, m00, m11});
}

inline DenseElementsAttr getMatrixS(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const auto& matrix = {1.0 + 0i, 0.0 + 0i,  // row 0
                        0.0 + 0i, 0.0 + 1i}; // row 1
  return DenseElementsAttr::get(type, matrix);
}

inline DenseElementsAttr getMatrixSdg(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const auto& matrix = {1.0 + 0i, 0.0 + 0i,  // row 0
                        0.0 + 0i, 0.0 - 1i}; // row 1
  return DenseElementsAttr::get(type, matrix);
}

inline DenseElementsAttr getMatrixT(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const std::complex<double> m00 = 1.0 + 0i;
  const std::complex<double> m01 = 0.0 + 0i;
  const std::complex<double> m11 = std::exp(1i * std::numbers::pi / 4.0);
  return DenseElementsAttr::get(type, {m00, m01, m01, m11});
}

inline DenseElementsAttr getMatrixTdg(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const std::complex<double> m00 = 1.0 + 0i;
  const std::complex<double> m01 = 0.0 + 0i;
  const std::complex<double> m11 = std::exp(-1i * std::numbers::pi / 4.0);
  return DenseElementsAttr::get(type, {m00, m01, m01, m11});
}

inline DenseElementsAttr getMatrixSX(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const std::complex<double> m00 = (1.0 + 1i) / 2.0;
  const std::complex<double> m01 = (1.0 - 1i) / 2.0;
  return DenseElementsAttr::get(type, {m00, m01, m01, m00});
}

inline DenseElementsAttr getMatrixSXdg(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const std::complex<double> m00 = (1.0 - 1i) / 2.0;
  const std::complex<double> m01 = (1.0 + 1i) / 2.0;
  return DenseElementsAttr::get(type, {m00, m01, m01, m00});
}

inline DenseElementsAttr getMatrixRX(MLIRContext* ctx, double theta) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const std::complex<double> m00 = std::cos(theta / 2.0) + 0i;
  const std::complex<double> m01 = -1i * std::sin(theta / 2.0);
  return DenseElementsAttr::get(type, {m00, m01, m01, m00});
}

inline DenseElementsAttr getMatrixRY(MLIRContext* ctx, double theta) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const std::complex<double> m00 = std::cos(theta / 2.0) + 0i;
  const std::complex<double> m01 = -std::sin(theta / 2.0) + 0i;
  return DenseElementsAttr::get(type, {m00, m01, -m01, m00});
}

inline DenseElementsAttr getMatrixRZ(MLIRContext* ctx, double theta) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const std::complex<double> m00 = std::exp(-1i * theta / 2.0);
  const std::complex<double> m01 = 0.0 + 0i;
  const std::complex<double> m11 = std::exp(1i * theta / 2.0);
  return DenseElementsAttr::get(type, {m00, m01, m01, m11});
}

inline DenseElementsAttr getMatrixP(MLIRContext* ctx, double theta) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const std::complex<double> m00 = 1.0 + 0i;
  const std::complex<double> m01 = 0.0 + 0i;
  const std::complex<double> m11 = std::exp(1i * theta);
  return DenseElementsAttr::get(type, {m00, m01, m01, m11});
}

inline DenseElementsAttr getMatrixU2(MLIRContext* ctx, double phi,
                                     double lambda) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const std::complex<double> m00 = 1.0 / std::sqrt(2) + 0i;
  const std::complex<double> m01 = -std::exp(1i * lambda) / std::sqrt(2);
  const std::complex<double> m10 = std::exp(1i * phi) / std::sqrt(2);
  const std::complex<double> m11 = std::exp(1i * (phi + lambda)) / std::sqrt(2);
  return DenseElementsAttr::get(type, {m00, m01, m10, m11});
}

inline DenseElementsAttr getMatrixU(MLIRContext* ctx, double theta, double phi,
                                    double lambda) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const std::complex<double> m00 = std::cos(theta / 2.0) + 0i;
  const std::complex<double> m01 =
      -std::exp(1i * lambda) * std::sin(theta / 2.0);
  const std::complex<double> m10 = std::exp(1i * phi) * std::sin(theta / 2.0);
  const std::complex<double> m11 =
      std::exp(1i * (phi + lambda)) * std::cos(theta / 2.0);
  return DenseElementsAttr::get(type, {m00, m01, m10, m11});
}

inline DenseElementsAttr getMatrixR(MLIRContext* ctx, double theta,
                                    double phi) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const std::complex<double> m00 = std::cos(theta / 2.0) + 0i;
  const std::complex<double> m01 =
      -1i * std::exp(-1i * phi) * std::sin(theta / 2.0);
  const std::complex<double> m10 =
      -1i * std::exp(1i * phi) * std::sin(theta / 2.0);
  const std::complex<double> m11 = std::cos(theta / 2.0) + 0i;
  return DenseElementsAttr::get(type, {m00, m01, m10, m11});
}

inline DenseElementsAttr getMatrixSWAP(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({4, 4}, complexType);
  const auto matrix = {1.0 + 0i, 0.0 + 0i, 0.0 + 0i, 0.0 + 0i,  // row 0
                       0.0 + 0i, 0.0 + 0i, 1.0 + 0i, 0.0 + 0i,  // row 1
                       0.0 + 0i, 1.0 + 0i, 0.0 + 0i, 0.0 + 0i,  // row 2
                       0.0 + 0i, 0.0 + 0i, 0.0 + 0i, 1.0 + 0i}; // row 3
  return DenseElementsAttr::get(type, matrix);
}

inline DenseElementsAttr getMatrixiSWAP(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({4, 4}, complexType);
  const auto matrix = {1.0 + 0i, 0.0 + 0i, 0.0 + 0i, 0.0 + 0i,  // row 0
                       0.0 + 0i, 0.0 + 0i, 0.0 + 1i, 0.0 + 0i,  // row 1
                       0.0 + 0i, 0.0 + 1i, 0.0 + 0i, 0.0 + 0i,  // row 2
                       0.0 + 0i, 0.0 + 0i, 0.0 + 0i, 1.0 + 0i}; // row 3
  return DenseElementsAttr::get(type, matrix);
}

inline DenseElementsAttr getMatrixDCX(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({4, 4}, complexType);
  const auto matrix = {1.0 + 0i, 0.0 + 0i, 0.0 + 0i, 0.0 + 0i,  // row 0
                       0.0 + 0i, 0.0 + 0i, 1.0 + 0i, 0.0 + 0i,  // row 1
                       0.0 + 0i, 0.0 + 0i, 0.0 + 0i, 1.0 + 0i,  // row 2
                       0.0 + 0i, 1.0 + 0i, 0.0 + 0i, 0.0 + 0i}; // row 3
  return DenseElementsAttr::get(type, matrix);
}

inline DenseElementsAttr getMatrixECR(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({4, 4}, complexType);
  const std::complex<double> m0 = 0.0 + 0i;
  const std::complex<double> m1 = 1.0 / std::sqrt(2) + 0i;
  const std::complex<double> mi = 0.0 + 1i / std::sqrt(2);
  const auto matrix = {m0,  m0,  m1, mi,  // row 0
                       m0,  m0,  mi, m1,  // row 1
                       m1,  -mi, m0, m0,  // row 2
                       -mi, m1,  m0, m0}; // row 3
  return DenseElementsAttr::get(type, matrix);
}

inline DenseElementsAttr getMatrixCtrl(mlir::MLIRContext* ctx,
                                       size_t numControls,
                                       mlir::DenseElementsAttr target) {
  // Get dimensions of target matrix
  const auto& targetType = llvm::dyn_cast<RankedTensorType>(target.getType());
  if (!targetType || targetType.getRank() != 2 ||
      targetType.getDimSize(0) != targetType.getDimSize(1)) {
    llvm::report_fatal_error("Invalid target matrix");
  }
  const auto targetDim = targetType.getDimSize(0);

  // Get values of target matrix
  const auto& targetMatrix = target.getValues<std::complex<double>>();

  // Define dimensions and type of output matrix
  const auto dim = static_cast<int64_t>(std::pow(2, numControls) * targetDim);
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({dim, dim}, complexType);

  // Allocate output matrix
  std::vector<std::complex<double>> matrix;
  matrix.reserve(dim * dim);

  // Fill output matrix
  for (int64_t i = 0; i < dim; ++i) {
    for (int64_t j = 0; j < dim; ++j) {
      if (i < (dim - targetDim) && j < (dim - targetDim)) {
        matrix.push_back((i == j) ? 1.0 : 0.0);
      } else if (i >= (dim - targetDim) && j >= (dim - targetDim)) {
        matrix.push_back(targetMatrix[(i - (dim - targetDim)) * targetDim +
                                      (j - (dim - targetDim))]);
      } else {
        matrix.push_back(0.0);
      }
    }
  }

  ArrayRef<std::complex<double>> matrixRef(matrix);
  return DenseElementsAttr::get(type, matrixRef);
}

} // namespace mlir::utils
