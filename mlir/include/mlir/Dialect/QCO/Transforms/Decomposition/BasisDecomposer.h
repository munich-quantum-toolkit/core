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

#include "UnitaryMatrices.h"
#include "WeylDecomposition.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/SmallVector.h>

#include <array>
#include <complex>
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::qco::decomposition {

/// Intermediate single-qubit ``2×2`` unitaries produced while expanding a
/// two-qubit basis decomposition.
using TwoQubitLocalUnitaryList = llvm::SmallVector<Matrix2x2, 8>;

/**
 * Result of a two-qubit basis decomposition expressed as raw single-qubit
 * factors interleaved with a fixed number of basis-gate (entangler) uses.
 *
 * The factors are stored in emission order. For `i` in `[0, numBasisUses)` the
 * pair `(singleQubitFactors[2*i], singleQubitFactors[2*i + 1])` is applied to
 * qubits `1` and `0` respectively, followed by one entangler. The final pair
 * `(singleQubitFactors[2*numBasisUses], singleQubitFactors[2*numBasisUses+1])`
 * is applied after the last entangler. The list therefore has length
 * `2 * (numBasisUses + 1)`.
 */
struct TwoQubitNativeDecomposition {
  /// Number of basis-gate (entangler) uses.
  std::uint8_t numBasisUses = 0;
  /// Single-qubit factors in emission order (see struct comment).
  TwoQubitLocalUnitaryList singleQubitFactors;
  /// Residual global phase (radians) not represented by factors/entanglers.
  double globalPhase = 0.0;
};

/**
 * Decomposer that must be initialized with a two-qubit basis gate that will
 * be used to generate a circuit equivalent to a canonical gate (RXX+RYY+RZZ).
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
  /**
   * Create decomposer that allows two-qubit decompositions based on the
   * specified entangler matrix.
   * This entangler will appear between 0 and 3 times in each decomposition.
   * The 4x4 matrix must be in MQT operand order (qubit 0 = MSB).
   */
  [[nodiscard]] static TwoQubitBasisDecomposer
  create(const Matrix4x4& basisMatrix, double basisFidelity);

  /**
   * Perform decomposition using the basis gate of this decomposer.
   *
   * @param targetDecomposition Prepared Weyl decomposition of unitary matrix
   *                            to be decomposed.
   * @param numBasisGateUses Force use of given number of basis gates. When
   *                         unset, the optimal count is selected from the
   *                         Hilbert-Schmidt traces.
   * @return The single-qubit factors and entangler count, or `std::nullopt`
   *         when more than one basis gate would be required but the basis gate
   *         is not super-controlled.
   */
  [[nodiscard]] std::optional<TwoQubitNativeDecomposition> twoQubitDecompose(
      const decomposition::TwoQubitWeylDecomposition& targetDecomposition,
      std::optional<std::uint8_t> numBasisGateUses) const;

protected:
  // NOLINTBEGIN(modernize-pass-by-value)
  /**
   * Constructs decomposer instance.
   */
  TwoQubitBasisDecomposer(
      double basisFidelity,
      const decomposition::TwoQubitWeylDecomposition& basisDecomposer,
      bool isSuperControlled, const Matrix2x2& u0l, const Matrix2x2& u0r,
      const Matrix2x2& u1l, const Matrix2x2& u1ra, const Matrix2x2& u1rb,
      const Matrix2x2& u2la, const Matrix2x2& u2lb, const Matrix2x2& u2ra,
      const Matrix2x2& u2rb, const Matrix2x2& u3l, const Matrix2x2& u3r,
      const Matrix2x2& q0l, const Matrix2x2& q0r, const Matrix2x2& q1la,
      const Matrix2x2& q1lb, const Matrix2x2& q1ra, const Matrix2x2& q1rb,
      const Matrix2x2& q2l, const Matrix2x2& q2r)
      : basisFidelity{basisFidelity}, basisDecomposer{basisDecomposer},
        isSuperControlled{isSuperControlled}, u0l{u0l}, u0r{u0r}, u1l{u1l},
        u1ra{u1ra}, u1rb{u1rb}, u2la{u2la}, u2lb{u2lb}, u2ra{u2ra}, u2rb{u2rb},
        u3l{u3l}, u3r{u3r}, q0l{q0l}, q0r{q0r}, q1la{q1la}, q1lb{q1lb},
        q1ra{q1ra}, q1rb{q1rb}, q2l{q2l}, q2r{q2r} {}
  // NOLINTEND(modernize-pass-by-value)

  /**
   * Calculate decompositions when no basis gate is required.
   *
   * Decompose target :math:`\sim U_d(x, y, z)` with 0 uses of the
   * basis gate. Result :math:`U_r` has trace:
   *
   * .. math::
   *
   *     \Big\vert\text{Tr}(U_r\cdot U_\text{target}^{\dag})\Big\vert =
   *     4\Big\vert (\cos(x)\cos(y)\cos(z)+ j \sin(x)\sin(y)\sin(z)\Big\vert
   *
   * which is optimal for all targets and bases
   */
  [[nodiscard]] static TwoQubitLocalUnitaryList
  decomp0(const decomposition::TwoQubitWeylDecomposition& target);

  /**
   * Calculate decompositions when one basis gate is required.
   *
   * Decompose target :math:`\sim U_d(x, y, z)` with 1 use of the
   * basis gate :math:`\sim U_d(a, b, c)`. Result :math:`U_r` has trace:
   *
   * .. math::
   *
   *     \Big\vert\text{Tr}(U_r \cdot U_\text{target}^{\dag})\Big\vert =
   *     4\Big\vert \cos(x-a)\cos(y-b)\cos(z-c) + j
   *     \sin(x-a)\sin(y-b)\sin(z-c)\Big\vert
   *
   * which is optimal for all targets and bases with ``z==0`` or ``c==0``.
   */
  [[nodiscard]] TwoQubitLocalUnitaryList
  decomp1(const decomposition::TwoQubitWeylDecomposition& target) const;

  /**
   * Calculate decompositions when two basis gates are required.
   *
   * Decompose target :math:`\sim U_d(x, y, z)` with 2 uses of the
   * basis gate.
   *
   * For supercontrolled basis :math:`\sim U_d(\pi/4, b, 0)`, all b, result
   * :math:`U_r` has trace
   *
   * .. math::
   *
   *     \Big\vert\text{Tr}(U_r \cdot U_\text{target}^\dag) \Big\vert =
   * 4\cos(z)
   *
   * which is the optimal approximation for basis of CNOT-class
   * :math:`\sim U_d(\pi/4, 0, 0)` or DCNOT-class
   * :math:`\sim U_d(\pi/4, \pi/4, 0)` and any target. It may be sub-optimal
   * for :math:`b \neq 0` (i.e. there exists an exact decomposition for any
   * target using :math:`B \sim U_d(\pi/4, \pi/8, 0)`, but it may not be this
   * decomposition). This is an exact decomposition for supercontrolled basis
   * and target :math:`\sim U_d(x, y, 0)`. No guarantees for
   * non-supercontrolled basis.
   */
  [[nodiscard]] TwoQubitLocalUnitaryList decomp2Supercontrolled(
      const decomposition::TwoQubitWeylDecomposition& target) const;

  /**
   * Calculate decompositions when three basis gates are required.
   *
   * Decompose target with 3 uses of the basis.
   *
   * This is an exact decomposition for supercontrolled basis
   * :math:`\sim U_d(\pi/4, b, 0)`, all b, and any target. No guarantees for
   * non-supercontrolled basis.
   */
  [[nodiscard]] TwoQubitLocalUnitaryList decomp3Supercontrolled(
      const decomposition::TwoQubitWeylDecomposition& target) const;

  /**
   * Calculate traces for a combination of the parameters of the canonical
   * gates of the target and basis decompositions.
   * This can be used to determine the smallest number of basis gates that are
   * necessary to construct an equivalent to the canonical gate.
   */
  [[nodiscard]] std::array<std::complex<double>, 4>
  traces(const decomposition::TwoQubitWeylDecomposition& target) const;

  [[nodiscard]] static bool relativeEq(double lhs, double rhs, double epsilon,
                                       double maxRelative);

private:
  // fidelity with which the basis gate decomposition has been calculated
  double basisFidelity;
  // cached decomposition for basis gate
  decomposition::TwoQubitWeylDecomposition basisDecomposer;
  // true if basis gate is super-controlled
  bool isSuperControlled;

  // pre-built components for decomposition with 3 basis gates
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

  // pre-built components for decomposition with 2 basis gates
  Matrix2x2 q0l;
  Matrix2x2 q0r;
  Matrix2x2 q1la;
  Matrix2x2 q1lb;
  Matrix2x2 q1ra;
  Matrix2x2 q1rb;
  Matrix2x2 q2l;
  Matrix2x2 q2r;
};

} // namespace mlir::qco::decomposition
