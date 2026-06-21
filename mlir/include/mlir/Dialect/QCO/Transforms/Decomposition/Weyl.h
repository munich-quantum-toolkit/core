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

#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/SmallVector.h>

#include <array>
#include <complex>
#include <cstdint>
#include <optional>

namespace mlir::qco::decomposition {

/**
 * @brief Weyl decomposition of a 2-qubit unitary matrix (4x4).
 *
 * The result consists of four 2x2 single-qubit matrices (`k1l`, `k2l`,
 * `k1r`, `k2r`) and three parameters for a canonical gate (`a`, `b`, `c`).
 * The canonical gate is `RXX(-2 * a) * RYY(-2 * b) * RZZ(-2 * c)`.
 *
 * @note Adapted from TwoQubitWeylDecomposition in the IBM Qiskit framework.
 *       (C) Copyright IBM 2023
 *
 *       This code is licensed under the Apache License, Version 2.0. You may
 *       obtain a copy of this license in the LICENSE.txt file in the root
 *       directory of this source tree or at
 *       https://www.apache.org/licenses/LICENSE-2.0.
 *
 *       Any modifications or derivative works of this code must retain this
 *       copyright notice, and modified files need to carry a notice
 *       indicating that they have been altered from the originals.
 */
class TwoQubitWeylDecomposition {
public:
  static TwoQubitWeylDecomposition create(const Matrix4x4& unitaryMatrix,
                                          std::optional<double> fidelity);

  ~TwoQubitWeylDecomposition() = default;
  TwoQubitWeylDecomposition() = default;
  TwoQubitWeylDecomposition(const TwoQubitWeylDecomposition&) = default;
  TwoQubitWeylDecomposition(TwoQubitWeylDecomposition&&) = default;
  TwoQubitWeylDecomposition&
  operator=(const TwoQubitWeylDecomposition&) = default;
  TwoQubitWeylDecomposition& operator=(TwoQubitWeylDecomposition&&) = default;

  [[nodiscard]] Matrix4x4 getCanonicalMatrix() const {
    return getCanonicalMatrix(a_, b_, c_);
  }

  [[nodiscard]] double a() const { return a_; }
  [[nodiscard]] double b() const { return b_; }
  [[nodiscard]] double c() const { return c_; }
  [[nodiscard]] double globalPhase() const { return globalPhase_; }

  /**
   * @brief Left single-qubit factor after the canonical gate.
   *
   * ```
   * q1 - k2r - C -  k1r  -
   *            A
   * q0 - k2l - N - *k1l* -
   * ```
   */
  [[nodiscard]] const Matrix2x2& k1l() const { return k1l_; }
  /**
   * @brief Left single-qubit factor before the canonical gate.
   *
   * ```
   * q1 -  k2r  - C - k1r -
   *              A
   * q0 - *k2l* - N - k1l -
   * ```
   */
  [[nodiscard]] const Matrix2x2& k2l() const { return k2l_; }
  /**
   * @brief Right single-qubit factor after the canonical gate.
   *
   * ```
   * q1 - k2r - C - *k1r* -
   *            A
   * q0 - k2l - N -  k1l  -
   * ```
   */
  [[nodiscard]] const Matrix2x2& k1r() const { return k1r_; }
  /**
   * @brief Right single-qubit factor before the canonical gate.
   *
   * ```
   * q1 - *k2r* - C - k1r -
   *              A
   * q0 -  k2l  - N - k1l -
   * ```
   */
  [[nodiscard]] const Matrix2x2& k2r() const { return k2r_; }

  [[nodiscard]] static Matrix4x4 getCanonicalMatrix(double a, double b,
                                                    double c);

private:
  bool applySpecialization();

  double a_{};
  double b_{};
  double c_{};
  double globalPhase_{};
  Matrix2x2 k1l_;
  Matrix2x2 k2l_;
  Matrix2x2 k1r_;
  Matrix2x2 k2r_;
  std::uint8_t specializationKind_{0};
  std::optional<double> requestedFidelity;
};

using TwoQubitLocalUnitaryList = llvm::SmallVector<Matrix2x2, 8>;

/**
 * @brief Result of a two-qubit basis decomposition as single-qubit factors and
 *        entangler uses.
 *
 * Factors are stored in emission order. For `i` in `[0, numBasisUses)` the
 * pair `(singleQubitFactors[2*i], singleQubitFactors[2*i + 1])` is applied to
 * qubits `1` and `0` respectively, followed by one entangler. The final pair
 * `(singleQubitFactors[2*numBasisUses], singleQubitFactors[2*numBasisUses+1])`
 * is applied after the last entangler. The list therefore has length
 * `2 * (numBasisUses + 1)`.
 */
struct TwoQubitNativeDecomposition {
  std::uint8_t numBasisUses = 0;
  TwoQubitLocalUnitaryList singleQubitFactors;
  double globalPhase = 0.0;
};

/**
 * @brief Decomposer initialized with a two-qubit basis gate for canonical-gate
 *        (RXX+RYY+RZZ) synthesis.
 *
 * @note Adapted from TwoQubitBasisDecomposer in the IBM Qiskit framework.
 *       (C) Copyright IBM 2023
 *
 *       This code is licensed under the Apache License, Version 2.0. You may
 *       obtain a copy of this license in the LICENSE.txt file in the root
 *       directory of this source tree or at
 *       https://www.apache.org/licenses/LICENSE-2.0.
 *
 *       Any modifications or derivative works of this code must retain this
 *       copyright notice, and modified files need to carry a notice
 *       indicating that they have been altered from the originals.
 */
class TwoQubitBasisDecomposer {
public:
  [[nodiscard]] static TwoQubitBasisDecomposer
  create(const Matrix4x4& basisMatrix, double basisFidelity);

  [[nodiscard]] std::optional<TwoQubitNativeDecomposition>
  twoQubitDecompose(const TwoQubitWeylDecomposition& targetDecomposition,
                    std::optional<std::uint8_t> numBasisGateUses) const;

private:
  struct SmbPrecomputed {
    Matrix2x2 u0l;
    Matrix2x2 u0r;
    Matrix2x2 u1l;
    Matrix2x2 u1ra;
    Matrix2x2 u1rb;
    Matrix2x2 u2la;
    Matrix2x2 u2lb;
    Matrix2x2 u2ra;
    Matrix2x2 u2rb;
    Matrix2x2 u3l;
    Matrix2x2 u3r;
    Matrix2x2 q0l;
    Matrix2x2 q0r;
    Matrix2x2 q1la;
    Matrix2x2 q1lb;
    Matrix2x2 q1ra;
    Matrix2x2 q1rb;
    Matrix2x2 q2l;
    Matrix2x2 q2r;
  };

  [[nodiscard]] TwoQubitLocalUnitaryList
  decomp0(const TwoQubitWeylDecomposition& target) const;
  [[nodiscard]] TwoQubitLocalUnitaryList
  decomp1(const TwoQubitWeylDecomposition& target) const;
  [[nodiscard]] TwoQubitLocalUnitaryList
  decomp2Supercontrolled(const TwoQubitWeylDecomposition& target) const;
  [[nodiscard]] TwoQubitLocalUnitaryList
  decomp3Supercontrolled(const TwoQubitWeylDecomposition& target) const;
  [[nodiscard]] std::array<std::complex<double>, 4>
  traces(const TwoQubitWeylDecomposition& target) const;

  double basisFidelity{};
  TwoQubitWeylDecomposition basisDecomposer;
  bool isSuperControlled{};
  SmbPrecomputed smb{};
};

} // namespace mlir::qco::decomposition
