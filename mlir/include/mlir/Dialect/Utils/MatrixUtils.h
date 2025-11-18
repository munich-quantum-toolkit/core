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

using namespace std::complex_literals;

namespace mlir::utils {

inline DenseElementsAttr getMatrixX(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const auto& matrix = {0.0 + 0i, 1.0 + 0i,  // row 0
                        1.0 + 0i, 0.0 + 0i}; // row 1
  return DenseElementsAttr::get(type, matrix);
}

inline DenseElementsAttr getMatrixRX(MLIRContext* ctx, double theta) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const std::complex<double> m0 = std::cos(theta / 2) + 0i;
  const std::complex<double> m1 = -1i * std::sin(theta / 2);
  return DenseElementsAttr::get(type, {m0, m1, m1, m0});
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

inline DenseElementsAttr getMatrixSWAP(MLIRContext* ctx) {
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({4, 4}, complexType);
  const auto matrix = {1.0 + 0i, 0.0 + 0i, 0.0 + 0i, 0.0 + 0i,  // row 0
                       0.0 + 0i, 0.0 + 0i, 1.0 + 0i, 0.0 + 0i,  // row 1
                       0.0 + 0i, 1.0 + 0i, 0.0 + 0i, 0.0 + 0i,  // row 2
                       0.0 + 0i, 0.0 + 0i, 0.0 + 0i, 1.0 + 0i}; // row 3
  return DenseElementsAttr::get(type, matrix);
}

inline DenseElementsAttr getMatrixCtrl(mlir::MLIRContext* ctx,
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
  const auto dim = 2 * targetDim;
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({dim, dim}, complexType);

  // Allocate output matrix
  std::vector<std::complex<double>> matrix;
  matrix.reserve(dim * dim);

  // Fill output matrix
  for (int64_t i = 0; i < dim; ++i) {
    for (int64_t j = 0; j < dim; ++j) {
      if (i < targetDim && j < targetDim) {
        matrix.push_back((i == j) ? 1.0 : 0.0);
      } else if (i >= targetDim && j >= targetDim) {
        matrix.push_back(
            targetMatrix[(i - targetDim) * targetDim + (j - targetDim)]);
      } else {
        matrix.push_back(0.0);
      }
    }
  }

  ArrayRef<std::complex<double>> matrixRef(matrix);
  return DenseElementsAttr::get(type, matrixRef);
}

} // namespace mlir::utils
