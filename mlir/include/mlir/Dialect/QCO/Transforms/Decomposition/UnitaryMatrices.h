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

#include "Gate.h"

#include <Eigen/Core>
#include <llvm/ADT/SmallVector.h>

/// Standard-basis matrix factories for the decomposition layer. Two-qubit
/// matrices use the same computational-basis index bit order as
/// ``UnitaryOpInterface::getUnitaryMatrix4x4`` (qubit 0 labels the high bit).

namespace mlir::qco::decomposition {

inline constexpr double FRAC1_SQRT2 =
    0.707106781186547524400844362104849039284835937688474036588L;

/// Generic 3-parameter single-qubit unitary `U(theta, phi, lambda)`.
[[nodiscard]] Eigen::Matrix2cd uMatrix(double theta, double phi, double lambda);
/// `U2(phi, lambda) == U(pi/2, phi, lambda)`.
[[nodiscard]] Eigen::Matrix2cd u2Matrix(double phi, double lambda);
/// Axis rotations `exp(-i theta/2 * sigma_{x,y,z})`.
[[nodiscard]] Eigen::Matrix2cd rxMatrix(double theta);
[[nodiscard]] Eigen::Matrix2cd ryMatrix(double theta);
[[nodiscard]] Eigen::Matrix2cd rzMatrix(double theta);
/// Two-qubit Ising-style rotations on the `XX`, `YY`, `ZZ` generators.
[[nodiscard]] Eigen::Matrix4cd rxxMatrix(double theta);
[[nodiscard]] Eigen::Matrix4cd ryyMatrix(double theta);
[[nodiscard]] Eigen::Matrix4cd rzzMatrix(double theta);
/// Phase gate `diag(1, exp(i lambda))`.
[[nodiscard]] Eigen::Matrix2cd pMatrix(double lambda);

inline const Eigen::Matrix2cd H_GATE{{FRAC1_SQRT2, FRAC1_SQRT2},
                                     {FRAC1_SQRT2, -FRAC1_SQRT2}};

/// Kronecker-embed a 2x2 on wire ``qubitId`` (identity on the other wire).
[[nodiscard]] Eigen::Matrix4cd
expandToTwoQubits(const Eigen::Matrix2cd& singleQubitMatrix, QubitId qubitId);

/// Construct the 2x2 / 4x4 matrix described by `gate`. Two-qubit gates are
/// returned in the convention matching `expandToTwoQubits` + the gate's own
/// operand order.
[[nodiscard]] Eigen::Matrix2cd getSingleQubitMatrix(const Gate& gate);
[[nodiscard]] Eigen::Matrix4cd getTwoQubitMatrix(const Gate& gate);

} // namespace mlir::qco::decomposition
