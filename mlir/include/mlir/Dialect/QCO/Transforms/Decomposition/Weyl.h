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

#include <mlir/Support/LLVM.h>

#include <array>
#include <complex>
#include <cstdint>
#include <optional>

namespace mlir::qco::decomposition {

/** Numeric tolerance for Weyl internal matrix checks. */
inline constexpr double WEYL_TOLERANCE = 100 * MATRIX_TOLERANCE;

/**
 * @brief Weyl decomposition of a 2-qubit unitary.
 *
 * A 4x4 unitary is factored as
 * `(K1l ⊗ K1r) · U_canon(a,b,c) · (K2l ⊗ K2r)` up to global phase, where
 * `U_canon(a,b,c) = RXX(-2a) · RYY(-2b) · RZZ(-2c)`.
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
  [[nodiscard]] static TwoQubitWeylDecomposition
  create(const Matrix4x4& unitaryMatrix, std::optional<double> fidelity);

  [[nodiscard]] Matrix4x4 getCanonicalMatrix() const {
    return getCanonicalMatrix(a_, b_, c_);
  }

  [[nodiscard]] double a() const { return a_; }
  [[nodiscard]] double b() const { return b_; }
  [[nodiscard]] double c() const { return c_; }
  [[nodiscard]] double globalPhase() const { return globalPhase_; }

  /** @brief Single-qubit factor on qubit 0 after the canonical gate. */
  [[nodiscard]] const Matrix2x2& k1l() const { return k1l_; }
  /** @brief Single-qubit factor on qubit 0 before the canonical gate. */
  [[nodiscard]] const Matrix2x2& k2l() const { return k2l_; }
  /** @brief Single-qubit factor on qubit 1 after the canonical gate. */
  [[nodiscard]] const Matrix2x2& k1r() const { return k1r_; }
  /** @brief Single-qubit factor on qubit 1 before the canonical gate. */
  [[nodiscard]] const Matrix2x2& k2r() const { return k2r_; }

  [[nodiscard]] static Matrix4x4 getCanonicalMatrix(double a, double b,
                                                    double c);

private:
  bool applySpecialization(const std::optional<double>& requestedFidelity);

  double a_{};
  double b_{};
  double c_{};
  double globalPhase_{};
  Matrix2x2 k1l_;
  Matrix2x2 k2l_;
  Matrix2x2 k1r_;
  Matrix2x2 k2r_;
};

/**
 * @brief Native two-qubit synthesis result.
 *
 * Emission order: for each `i`, apply `singleQubitFactors[2*i+1]` on q0 and
 * `singleQubitFactors[2*i]` on q1, then the basis gate (except after the last
 * pair).
 */
struct TwoQubitNativeDecomposition {
  std::uint8_t numBasisUses = 0;
  SmallVector<Matrix2x2, 8> singleQubitFactors;
  double globalPhase = 0.0;
};

/**
 * @brief Decomposer for a fixed two-qubit basis gate (e.g. CX/CZ).
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

  [[nodiscard]] static SmallVector<Matrix2x2, 8>
  decomp0(const TwoQubitWeylDecomposition& target);
  [[nodiscard]] SmallVector<Matrix2x2, 8>
  decomp1(const TwoQubitWeylDecomposition& target) const;
  [[nodiscard]] SmallVector<Matrix2x2, 8>
  decomp2Supercontrolled(const TwoQubitWeylDecomposition& target) const;
  [[nodiscard]] SmallVector<Matrix2x2, 8>
  decomp3Supercontrolled(const TwoQubitWeylDecomposition& target) const;
  [[nodiscard]] std::array<std::complex<double>, 4>
  traces(const TwoQubitWeylDecomposition& target) const;

  double basisFidelity{};
  TwoQubitWeylDecomposition basisDecomposer;
  bool isSuperControlled{};
  SmbPrecomputed smb{};
};

/** @brief Weyl-decomposes @p target using @p basisMatrix as entangler. */
[[nodiscard]] std::optional<TwoQubitNativeDecomposition>
decomposeTwoQubitWithBasis(
    const Matrix4x4& target, const Matrix4x4& basisMatrix,
    double basisFidelity = 1.0,
    std::optional<std::uint8_t> numBasisUses = std::nullopt);

} // namespace mlir::qco::decomposition
