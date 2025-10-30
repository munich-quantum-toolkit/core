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
#include <llvm/ADT/ArrayRef.h>

using namespace std::complex_literals;

namespace mlir::utils {

inline llvm::ArrayRef<std::complex<double>> getMatrixX() {
  return {0.0, 1.0,  // row 0
          1.0, 0.0}; // row 1
}

inline llvm::ArrayRef<std::complex<double>> getMatrixRX(double theta) {
  const std::complex<double> m0(std::cos(theta / 2), 0);
  const std::complex<double> m1(0, -std::sin(theta / 2));
  return {m0, m1, m1, m0};
}

inline llvm::ArrayRef<std::complex<double>> getMatrixU2(double phi,
                                                        double lambda) {
  const std::complex<double> m00(1.0, 0.0);
  const std::complex<double> m01 = -std::exp(1i * lambda);
  const std::complex<double> m10 = -std::exp(1i * phi);
  const std::complex<double> m11 = std::exp(1i * (phi + lambda));
  return {m00, m01, m10, m11};
}

inline llvm::ArrayRef<std::complex<double>> getMatrixSWAP() {
  return {1.0, 0.0, 0.0, 0.0,  // row 0
          0.0, 0.0, 1.0, 0.0,  // row 1
          0.0, 1.0, 0.0, 0.0,  // row 2
          0.0, 0.0, 0.0, 1.0}; // row 3
}

} // namespace mlir::utils
