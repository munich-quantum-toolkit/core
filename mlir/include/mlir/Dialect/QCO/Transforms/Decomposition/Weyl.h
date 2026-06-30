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
 *       (C) Copyright IBM 2026
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

  void finalizeSpecializationPhase(bool flippedFromOriginal,
                                   double preSpecializationA,
                                   double preSpecializationB,
                                   double preSpecializationC,
                                   const std::optional<double>& fidelity);

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
 *       (C) Copyright IBM 2026
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
  /**
   * @brief Precomputes basis-gate data for repeated target decompositions.
   *
   * Creation performs a full Weyl decomposition of @p basisMatrix and
   * precomputes the single-qubit templates used by @ref twoQubitDecompose.
   * Reuse one instance for many targets that share the same entangler
   * (e.g. CX) via @ref decomposeTarget.
   */
  [[nodiscard]] static TwoQubitBasisDecomposer
  create(const Matrix4x4& basisMatrix, double basisFidelity);

  /**
   * @brief Decomposes a target Weyl decomposition into single-qubit factors and
   *        basis-gate uses.
   *
   * @param targetDecomposition Weyl decomposition of the target unitary.
   * @param numBasisGateUses Requested number of basis-gate applications in
   *        `{0, 1, 2, 3}`. Pass `std::nullopt` to pick the count that
   *        maximizes `traceToFidelity(trace[i]) * basisFidelity^i` over
   *        `i ∈ {0, 1, 2, 3}`.
   * @return A native decomposition on success, or `std::nullopt` when the
   *         requested count is unsupported (e.g. more than one basis gate for a
   *         non-super-controlled basis, or a value outside `{0, 1, 2, 3}`).
   */
  [[nodiscard]] std::optional<TwoQubitNativeDecomposition>
  twoQubitDecompose(const TwoQubitWeylDecomposition& targetDecomposition,
                    std::optional<std::uint8_t> numBasisGateUses) const;

  /**
   * @brief Decomposes @p targetUnitary using this cached basis decomposer.
   *
   * Only the target undergoes Weyl decomposition; basis precomputation from
   * @ref create is reused.
   */
  [[nodiscard]] std::optional<TwoQubitNativeDecomposition> decomposeTarget(
      const Matrix4x4& targetUnitary,
      std::optional<std::uint8_t> numBasisGateUses = std::nullopt) const;

private:
  // clang-format off
  /**
   * @brief Precomputed single-qubit templates for super-controlled basis
   * synthesis.
   *
   * Populated once in @ref create from the Weyl decomposition of the basis
   * gate. Members are combined with target Weyl factors (`K1l`, `K1r`, `K2l`,
   * `K2r`) and parameterized `RZ` rotations in @ref decomp2Supercontrolled and
   * @ref decomp3Supercontrolled.
   *
   * Naming: suffix `l` / `r` is the q0 / q1 factor in `kron(q0_factor,
   * q1_factor)`. Pairs `*la` / `*ra` and `*lb` / `*rb` sandwich an `RZ` on that
   * wire
   * (`u1ra·RZ(-2c)·u1rb`, `u2la·RZ(-2a)·u2lb`, `u2ra·RZ(2b)·u2rb`, etc.).
   * `u3*` / `u2*` / `u1*` / `u0*` index layers from outside (post-`K2`) to
   * inside (pre-`K1`) in the three-basis layout; `q*` members are the
   * two-basis-only substitutes for the inner `u0*` and the `u2*a` halves of
   * layer 1. `u3*`, `u2l*b`, and `u2r*b` are reused in both decomp paths
   * (formerly duplicated as `q2*` / `q1l*b`).
   *
   * Emission order matches @ref TwoQubitNativeDecomposition: layer `i` applies
   * `kron(factors[2*i+1], factors[2*i])`, then the basis gate `E` (except after
   * the last layer). `E` is the fixed basis entangler (e.g. CX).
   *
   * @verbatim
   * Two basis gates (numBasisUses = 2); left = outer, right = inner:
   *
   *          +---------+         +-------------------+         +---------+
   *     q_0: | u3l.K2l |----+----| q1la.RZ(-2a).u2lb |----+----| K1l.q0l |
   *          +---------+    |    +-------------------+    |    +---------+
   *                         E                             E
   *          +---------+    |    +-------------------+    |    +---------+
   *     q_1: | u3r.K2r |----+----| q1ra.RZ(2b).u2rb  |----+----| K1r.q0r |
   *          +---------+         +-------------------+         +---------+
   *
   * Three basis gates (numBasisUses = 3):
   *
   *          +---------+         +-------------------+         +-------------------+         +---------+
   *     q_0: | u3l.K2l |----+----| u2la.RZ(-2a).u2lb |----+----|        u1l        |----+----| K1l.u0l |
   *          +---------+    |    +-------------------+    |    +-------------------+    |    +---------+
   *                         E                             E                             E
   *          +---------+    |    +-------------------+    |    +-------------------+    |    +---------+
   *     q_1: | u3r.K2r |----+----| u2ra.RZ(2b).u2rb  |----+----| u1ra.RZ(-2c).u1rb |----+----| K1r.u0r |
   *          +---------+         +-------------------+         +-------------------+         +---------+
   * @endverbatim
   */
  // clang-format on
  struct SmbPrecomputed {
    Matrix2x2 u0l;  ///< Inner q0 template (3-basis); combined with `K1l`.
    Matrix2x2 u0r;  ///< Inner q1 template (3-basis); combined with `K1r`.
    Matrix2x2 u1l;  ///< Middle-layer q0 factor (3-basis only).
    Matrix2x2 u1ra; ///< q1 factor before `RZ(-2c)` (3-basis middle layer).
    Matrix2x2 u1rb; ///< q1 factor after `RZ(-2c)` (3-basis middle layer).
    Matrix2x2 u2la; ///< q0 factor before `RZ(-2a)`.
    Matrix2x2 u2lb; ///< q0 factor after `RZ(-2a)`; shared with 2-basis path.
    Matrix2x2 u2ra; ///< q1 factor before `RZ(2b)` (3-basis only).
    Matrix2x2 u2rb; ///< q1 factor after `RZ(2b)`; shared with 2-basis path.
    Matrix2x2 u3l;  ///< Outermost q0 template; combined with `K2l`; shared with
                    ///< 2-basis path.
    Matrix2x2 u3r;  ///< Outermost q1 template; combined with `K2r`; shared with
                    ///< 2-basis path.
    Matrix2x2 q0l;  ///< Inner q0 template (2-basis); combined with `K1l`.
    Matrix2x2 q0r;  ///< Inner q1 template (2-basis); combined with `K1r`.
    Matrix2x2 q1la; ///< q0 factor before `RZ(-2a)` (2-basis layer 1).
    Matrix2x2 q1ra; ///< q1 factor before `RZ(2b)` (2-basis layer 1).
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

  TwoQubitBasisDecomposer() = default;

  double basisFidelity{};
  TwoQubitWeylDecomposition basisWeyl;
  bool isSuperControlled{};
  SmbPrecomputed smb{};
};

/**
 * @brief Convenience wrapper that builds a fresh basis decomposer per call.
 *
 * For a fixed basis gate decomposed many times, prefer caching
 * `TwoQubitBasisDecomposer::create(basisMatrix, basisFidelity)` and calling
 * `TwoQubitBasisDecomposer::decomposeTarget` for each target.
 */
[[nodiscard]] std::optional<TwoQubitNativeDecomposition>
decomposeTwoQubitWithBasis(
    const Matrix4x4& target, const Matrix4x4& basisMatrix,
    double basisFidelity = 1.0,
    std::optional<std::uint8_t> numBasisUses = std::nullopt);

} // namespace mlir::qco::decomposition
