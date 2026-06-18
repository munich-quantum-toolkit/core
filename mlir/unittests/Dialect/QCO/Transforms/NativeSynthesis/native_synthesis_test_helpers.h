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

#include "TestCaseUtils.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>

#include <complex>
#include <cstddef>
#include <optional>
#include <vector>

namespace mlir::qco::native_synth_test {

using mqt::test::isEquivalentUpToGlobalPhase;

/// Minimal dense, row-major, square complex matrix with runtime dimension.
///
/// Used by the multi-qubit equivalence checks (the synthesized circuits may
/// span more than two wires, so the fixed-size `Matrix2x2`/`Matrix4x4` are not
/// enough). Provides exactly the surface
/// `mqt::test::isEquivalentUpToGlobalPhase` needs: `adjoint()`, `operator*`,
/// scalar multiply, `trace()`, and `isApprox()`.
class TestMatrix {
public:
  TestMatrix() = default;
  explicit TestMatrix(std::size_t dim)
      : dim_(dim), data_(dim * dim, std::complex<double>{0.0, 0.0}) {}

  /// Identity matrix of dimension @p dim.
  [[nodiscard]] static TestMatrix identity(std::size_t dim);
  /// Promote a fixed `2×2` matrix to a `TestMatrix`.
  [[nodiscard]] static TestMatrix fromMatrix2x2(const Matrix2x2& matrix);
  /// Promote a fixed `4×4` matrix to a `TestMatrix`.
  [[nodiscard]] static TestMatrix fromMatrix4x4(const Matrix4x4& matrix);

  [[nodiscard]] std::size_t dim() const { return dim_; }

  [[nodiscard]] std::complex<double>& operator()(std::size_t row,
                                                 std::size_t col) {
    return data_[(row * dim_) + col];
  }
  [[nodiscard]] std::complex<double> operator()(std::size_t row,
                                                std::size_t col) const {
    return data_[(row * dim_) + col];
  }

  /// Matrix product (dimensions must match).
  [[nodiscard]] TestMatrix operator*(const TestMatrix& rhs) const;
  /// Element-wise scaling by a complex scalar.
  [[nodiscard]] TestMatrix operator*(std::complex<double> scalar) const;
  /// Conjugate transpose.
  [[nodiscard]] TestMatrix adjoint() const;
  /// Sum of diagonal entries.
  [[nodiscard]] std::complex<double> trace() const;
  /// Entry-wise approximate equality (false on dimension mismatch).
  [[nodiscard]] bool isApprox(const TestMatrix& other,
                              double tol = 1e-10) const;

private:
  std::size_t dim_ = 0;
  std::vector<std::complex<double>> data_;
};

/// Left scalar multiply, mirroring the right multiply above.
[[nodiscard]] inline TestMatrix operator*(std::complex<double> scalar,
                                          const TestMatrix& matrix) {
  return matrix * scalar;
}

[[nodiscard]] std::complex<double> phasedAmplitude(double magnitude,
                                                   double phase);
[[nodiscard]] Matrix2x2 u3Matrix(double theta, double phi, double lambda);
[[nodiscard]] bool isUnitary(const Matrix2x2& matrix, double atol = 1e-10);
[[nodiscard]] std::optional<double> evaluateConstF64(Value value);
bool extractSingleQubitMatrix(qco::UnitaryOpInterface op, Matrix2x2& out);
bool extractTwoQubitMatrix(qco::UnitaryOpInterface op, Matrix4x4& out);
[[nodiscard]] std::optional<Matrix4x4>
computeTwoQubitUnitaryFromModule(const OwningOpRef<ModuleOp>& moduleOp);
[[nodiscard]] TestMatrix expandOneQToN(const Matrix2x2& matrix, std::size_t q,
                                       std::size_t numQubits);
[[nodiscard]] TestMatrix expandTwoQToN(const Matrix4x4& matrix, std::size_t q0,
                                       std::size_t q1, std::size_t numQubits);
[[nodiscard]] std::optional<TestMatrix>
computeNQubitUnitaryFromModule(const OwningOpRef<ModuleOp>& moduleOp,
                               std::size_t maxQubits = 6);

} // namespace mlir::qco::native_synth_test
