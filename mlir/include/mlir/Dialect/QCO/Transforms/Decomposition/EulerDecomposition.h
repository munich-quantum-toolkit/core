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

#include <Eigen/Core>

#include <array>
#include <optional>

namespace mlir::qco::decomposition {

/**
 * Decompose a single-qubit unitary into a selected Euler-style gate basis.
 *
 * The returned sequence tracks both the emitted gates and the scalar phase
 * needed to reconstruct the input matrix exactly. This is stronger than the
 * usual "up to global phase" contract and is relied on by downstream
 * canonicalization and testing code.
 */
class EulerDecomposition {
public:
  /**
   * Decompose a 2x2 unitary into the gate alphabet described by
   * `targetBasis`.
   *
   * When `simplify` is true, near-zero angles are removed using `atol` (or
   * `DEFAULT_ATOL` if no override is provided). The returned global phase keeps
   * the decomposition exactly equal to `unitaryMatrix`.
   */
  [[nodiscard]] static OneQubitGateSequence
  generateCircuit(EulerBasis targetBasis, const Eigen::Matrix2cd& unitaryMatrix,
                  bool simplify, std::optional<double> atol);

  /**
   * Extract canonical Euler parameters for `matrix` in the requested basis.
   *
   * Some target bases reuse the same parameter extraction routine and differ
   * only during circuit emission. The returned array always contains
   * `(theta, phi, lambda, phase)` in this order.
   */
  [[nodiscard]] static std::array<double, 4>
  anglesFromUnitary(const Eigen::Matrix2cd& matrix, EulerBasis basis);

private:
  /// Extract parameters for a `rz(phi) · ry(theta) · rz(lambda)` factorization.
  [[nodiscard]] static std::array<double, 4>
  paramsZyz(const Eigen::Matrix2cd& matrix);

  /// Extract parameters for a `rz(phi) · rx(theta) · rz(lambda)` factorization.
  [[nodiscard]] static std::array<double, 4>
  paramsZxz(const Eigen::Matrix2cd& matrix);

  /// Extract parameters for a `rx(phi) · ry(theta) · rx(lambda)` factorization.
  [[nodiscard]] static std::array<double, 4>
  paramsXyx(const Eigen::Matrix2cd& matrix);

  /// Extract parameters for a `rx(phi) · rz(theta) · rx(lambda)` factorization.
  [[nodiscard]] static std::array<double, 4>
  paramsXzx(const Eigen::Matrix2cd& matrix);

  /**
   * Extract parameters for a `u1`/`p` + `sx` factorization.
   *
   * The returned angles are identical to `paramsZyz` but the phase is shifted
   * by `-0.5 * (theta + phi + lambda)` so that the `rz`/`sx` circuits emitted
   * by `decomposePsxGen` match the input matrix exactly (not only up to a
   * global phase).
   *
   * @note Adapted from `params_u1x_inner` in the IBM Qiskit framework.
   *       (C) Copyright IBM 2022
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
  [[nodiscard]] static std::array<double, 4>
  paramsU1x(const Eigen::Matrix2cd& matrix);

  /**
   * Emit a K-A-K circuit from already extracted Euler parameters.
   *
   * `kGate` is used for the outer rotations and `aGate` for the middle
   * rotation.
   *
   * @note Adapted from circuit_kak() in the IBM Qiskit framework.
   *       (C) Copyright IBM 2022
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
  [[nodiscard]] static OneQubitGateSequence
  decomposeKAK(double theta, double phi, double lambda, double phase,
               GateKind kGate, GateKind aGate, bool simplify,
               std::optional<double> atol);

  /**
   * Emit an `rz`/`sx`-style circuit for the `ZSX` and `ZSXX` bases.
   *
   * The emitted sequence is structurally identical to the one produced by
   * Qiskit's `circuit_psx_gen`. When `simplify` is enabled the number of `sx`
   * gates shrinks based on `theta`: zero `sx` gates for `theta ~= 0`, one
   * `sx` gate for `theta ~= pi/2`, and two `sx` gates otherwise.
   *
   * When `allowXShortcut` is true (i.e. for `ZSXX`), the general-case 2-`sx`
   * path additionally collapses `sx · rz(+/- pi) · sx` into a single `x`
   * gate when the middle rotation is congruent to +/- pi modulo 2 pi.
   *
   * @note Adapted from `circuit_psx_gen` in the IBM Qiskit framework.
   *       (C) Copyright IBM 2022
   *
   *       This code is licensed under the Apache License, Version 2.0. You
   *       may obtain a copy of this license in the LICENSE.txt file in the
   *       root directory of this source tree or at
   *       https://www.apache.org/licenses/LICENSE-2.0.
   *
   *       Any modifications or derivative works of this code must retain
   *       this copyright notice, and modified files need to carry a notice
   *       indicating that they have been altered from the originals.
   */
  [[nodiscard]] static OneQubitGateSequence
  decomposePsxGen(double theta, double phi, double lambda, double phase,
                  bool allowXShortcut, bool simplify,
                  std::optional<double> atol);
};
} // namespace mlir::qco::decomposition
