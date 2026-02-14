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

#include "EulerBasis.h"
#include "GateSequence.h"
#include "WeylDecomposition.h"

#include <Eigen/Core>
#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <optional>
#include <utility>

namespace mlir::qco::decomposition {

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
 *       http://www.apache.org/licenses/LICENSE-2.0.
 *
 *       Any modifications or derivative works of this code must retain this
 *       copyright notice, and modified files need to carry a notice
 *       indicating that they have been altered from the originals.
 */
class TwoQubitBasisDecomposer {
public:
  /**
   * Create decomposer that allows two-qubit decompositions based on the
   * specified basis gate.
   * This basis gate will appear between 0 and 3 times in each decompositions.
   * The order of qubits is relevant and will change the results accordingly.
   * The decomposer cannot handle different basis gates in the same
   * decomposition (different order of the qubits also counts as a different
   * basis gate).
   */
  [[nodiscard]] static TwoQubitBasisDecomposer create(const Gate& basisGate,
                                                      double basisFidelity);

  /**
   * Perform decomposition using the basis gate of this decomposer.
   *
   * @param targetDecomposition Prepared Weyl decomposition of unitary matrix
   *                            to be decomposed.
   * @param target1qEulerBases List of euler bases that should be tried out to
   *                           find the best one for each euler decomposition.
   *                           All bases will be mixed to get the best overall
   *                           result.
   * @param basisFidelity Fidelity for lowering the number of basis gates
   *                      required
   * @param approximate If true, use basisFidelity or, if std::nullopt, use
   *                    basisFidelity of this decomposer. If false, fidelity
   *                    of 1.0 will be assumed.
   * @param numBasisGateUses Force use of given number of basis gates.
   */
  [[nodiscard]] std::optional<TwoQubitGateSequence> twoQubitDecompose(
      const decomposition::TwoQubitWeylDecomposition& targetDecomposition,
      const llvm::SmallVector<EulerBasis>& target1qEulerBases,
      std::optional<double> basisFidelity, bool approximate,
      std::optional<std::uint8_t> numBasisGateUses) const;

protected:
  // NOLINTBEGIN(modernize-pass-by-value)
  /**
   * Constructs decomposer instance.
   */
  TwoQubitBasisDecomposer(
      Gate basisGate, double basisFidelity,
      const decomposition::TwoQubitWeylDecomposition& basisDecomposer,
      bool isSuperControlled, const Eigen::Matrix2cd& u0l,
      const Eigen::Matrix2cd& u0r, const Eigen::Matrix2cd& u1l,
      const Eigen::Matrix2cd& u1ra, const Eigen::Matrix2cd& u1rb,
      const Eigen::Matrix2cd& u2la, const Eigen::Matrix2cd& u2lb,
      const Eigen::Matrix2cd& u2ra, const Eigen::Matrix2cd& u2rb,
      const Eigen::Matrix2cd& u3l, const Eigen::Matrix2cd& u3r,
      const Eigen::Matrix2cd& q0l, const Eigen::Matrix2cd& q0r,
      const Eigen::Matrix2cd& q1la, const Eigen::Matrix2cd& q1lb,
      const Eigen::Matrix2cd& q1ra, const Eigen::Matrix2cd& q1rb,
      const Eigen::Matrix2cd& q2l, const Eigen::Matrix2cd& q2r)
      : basisGate{std::move(basisGate)}, basisFidelity{basisFidelity},
        basisDecomposer{basisDecomposer}, isSuperControlled{isSuperControlled},
        u0l{u0l}, u0r{u0r}, u1l{u1l}, u1ra{u1ra}, u1rb{u1rb}, u2la{u2la},
        u2lb{u2lb}, u2ra{u2ra}, u2rb{u2rb}, u3l{u3l}, u3r{u3r}, q0l{q0l},
        q0r{q0r}, q1la{q1la}, q1lb{q1lb}, q1ra{q1ra}, q1rb{q1rb}, q2l{q2l},
        q2r{q2r} {}
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
   *
   * @note The inline storage of llvm::SmallVector must be set to 0 to ensure
   *       correct Eigen alignment via heap allocation
   */
  [[nodiscard]] static llvm::SmallVector<Eigen::Matrix2cd, 0>
  decomp0(const decomposition::TwoQubitWeylDecomposition& target);

  /**
   * Calculate decompositions when one basis gate is required.
   *
   * Decompose target :math:`\sim U_d(x, y, z)` with 1 use of the
   * basis gate math:`\sim U_d(a, b, c)`. Result :math:`U_r` has trace:
   *
   * .. math::
   *
   *     \Big\vert\text{Tr}(U_r \cdot U_\text{target}^{\dag})\Big\vert =
   *     4\Big\vert \cos(x-a)\cos(y-b)\cos(z-c) + j
   *     \sin(x-a)\sin(y-b)\sin(z-c)\Big\vert
   *
   * which is optimal for all targets and bases with ``z==0`` or ``c==0``.
   *
   * @note The inline storage of llvm::SmallVector must be set to 0 to ensure
   *       correct Eigen alignment via heap allocation
   */
  [[nodiscard]] llvm::SmallVector<Eigen::Matrix2cd, 0>
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
   *
   * @note The inline storage of llvm::SmallVector must be set to 0 to ensure
   *       correct Eigen alignment via heap allocation
   */
  [[nodiscard]] llvm::SmallVector<Eigen::Matrix2cd, 0> decomp2Supercontrolled(
      const decomposition::TwoQubitWeylDecomposition& target) const;

  /**
   * Calculate decompositions when three basis gates are required.
   *
   * Decompose target with 3 uses of the basis.
   *
   * This is an exact decomposition for supercontrolled basis
   * :math:`\sim U_d(\pi/4, b, 0)`, all b, and any target. No guarantees for
   * non-supercontrolled basis.
   *
   * @note The inline storage of llvm::SmallVector must be set to 0 to ensure
   *       correct Eigen alignment via heap allocation
   */
  [[nodiscard]] llvm::SmallVector<Eigen::Matrix2cd, 0> decomp3Supercontrolled(
      const decomposition::TwoQubitWeylDecomposition& target) const;

  /**
   * Calculate traces for a combination of the parameters of the canonical
   * gates of the target and basis decompositions.
   * This can be used to determine the smallest number of basis gates that are
   * necessary to construct an equivalent to the canonical gate.
   */
  [[nodiscard]] std::array<std::complex<double>, 4>
  traces(const decomposition::TwoQubitWeylDecomposition& target) const;
  /**
   * Decompose a single-qubit unitary matrix into a single-qubit gate
   * sequence. Multiple euler bases may be specified and the one with the
   * least complexity will be chosen.
   */
  [[nodiscard]] static OneQubitGateSequence
  unitaryToGateSequence(const Eigen::Matrix2cd& unitaryMat,
                        const llvm::SmallVector<EulerBasis>& targetBasisList,
                        QubitId /*qubit*/,
                        // TODO: add error map here: per qubit a mapping of
                        // operation to error value for better calculateError()
                        bool simplify, std::optional<double> atol);

  [[nodiscard]] static bool relativeEq(double lhs, double rhs, double epsilon,
                                       double maxRelative);

private:
  // basis gate of this decomposer instance
  Gate basisGate{};
  // fidelity with which the basis gate decomposition has been calculated
  double basisFidelity;
  // cached decomposition for basis gate
  decomposition::TwoQubitWeylDecomposition basisDecomposer;
  // true if basis gate is super-controlled
  bool isSuperControlled;

  // pre-built components for decomposition with 3 basis gates
  Eigen::Matrix2cd u0l;
  Eigen::Matrix2cd u0r;
  Eigen::Matrix2cd u1l;
  Eigen::Matrix2cd u1ra;
  Eigen::Matrix2cd u1rb;
  Eigen::Matrix2cd u2la;
  Eigen::Matrix2cd u2lb;
  Eigen::Matrix2cd u2ra;
  Eigen::Matrix2cd u2rb;
  Eigen::Matrix2cd u3l;
  Eigen::Matrix2cd u3r;

  // pre-built components for decomposition with 2 basis gates
  Eigen::Matrix2cd q0l;
  Eigen::Matrix2cd q0r;
  Eigen::Matrix2cd q1la;
  Eigen::Matrix2cd q1lb;
  Eigen::Matrix2cd q1ra;
  Eigen::Matrix2cd q1rb;
  Eigen::Matrix2cd q2l;
  Eigen::Matrix2cd q2r;
};

} // namespace mlir::qco::decomposition
