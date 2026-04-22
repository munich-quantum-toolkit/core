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

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"

#include <Eigen/Core>
#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>

#include <complex>
#include <cstddef>
#include <optional>

namespace mlir::qco::native_synth_test {

template <typename Matrix>
bool isEquivalentUpToGlobalPhase(const Matrix& lhs, const Matrix& rhs,
                                 double atol = 1e-10) {
  const auto overlap = (rhs.adjoint() * lhs).trace();
  if (std::abs(overlap) <= atol) {
    return false;
  }
  const auto factor = overlap / std::abs(overlap);
  return lhs.isApprox(factor * rhs, atol);
}

[[nodiscard]] std::complex<double> phasedAmplitude(double magnitude,
                                                   double phase);
[[nodiscard]] Eigen::Matrix2cd u3Matrix(double theta, double phi,
                                        double lambda);
[[nodiscard]] bool isUnitary(const Eigen::Matrix2cd& m, double atol = 1e-10);
[[nodiscard]] std::optional<double> evaluateConstF64(Value value);
bool extractSingleQubitMatrix(qco::UnitaryOpInterface op,
                              Eigen::Matrix2cd& out);
bool extractTwoQubitMatrix(qco::UnitaryOpInterface op, Eigen::Matrix4cd& out);
[[nodiscard]] std::optional<Eigen::Matrix4cd>
computeTwoQubitUnitaryFromModule(const OwningOpRef<ModuleOp>& moduleOp);
[[nodiscard]] Eigen::MatrixXcd
expandOneQToN(const Eigen::Matrix2cd& m, std::size_t q, std::size_t numQubits);
[[nodiscard]] Eigen::MatrixXcd expandTwoQToN(const Eigen::Matrix4cd& m,
                                             std::size_t q0, std::size_t q1,
                                             std::size_t numQubits);
[[nodiscard]] std::optional<Eigen::MatrixXcd>
computeNQubitUnitaryFromModule(const OwningOpRef<ModuleOp>& moduleOp,
                               std::size_t maxQubits = 6);

} // namespace mlir::qco::native_synth_test
