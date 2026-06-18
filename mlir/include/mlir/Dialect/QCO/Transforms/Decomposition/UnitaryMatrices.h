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
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/SmallVector.h>

/// Standard-basis matrix factories for the decomposition layer. Two-qubit
/// matrices use the same computational-basis index bit order as
/// ``UnitaryOpInterface::getUnitaryMatrix4x4`` (qubit 0 labels the high bit).

namespace mlir::qco::decomposition {

inline constexpr double FRAC1_SQRT2 =
    0.707106781186547524400844362104849039284835937688474036588L;

/// Generic 3-parameter single-qubit unitary `U(theta, phi, lambda)`.
[[nodiscard]] Matrix2x2 uMatrix(double theta, double phi, double lambda);
/// `U2(phi, lambda) == U(pi/2, phi, lambda)`.
[[nodiscard]] Matrix2x2 u2Matrix(double phi, double lambda);
/// Axis rotations `exp(-i theta/2 * sigma_{x,y,z})`.
[[nodiscard]] Matrix2x2 rxMatrix(double theta);
[[nodiscard]] Matrix2x2 ryMatrix(double theta);
[[nodiscard]] Matrix2x2 rzMatrix(double theta);
/// Two-qubit Ising-style rotations on the `XX`, `YY`, `ZZ` generators.
[[nodiscard]] Matrix4x4 rxxMatrix(double theta);
[[nodiscard]] Matrix4x4 ryyMatrix(double theta);
[[nodiscard]] Matrix4x4 rzzMatrix(double theta);
/// Phase gate `diag(1, exp(i lambda))`.
[[nodiscard]] Matrix2x2 pMatrix(double lambda);

/// `SWAP` gate (4x4).
[[nodiscard]] const Matrix4x4& swapGate();
/// Hadamard gate (2x2).
[[nodiscard]] const Matrix2x2& hGate();
/// `i * sigma_z`; useful when factoring Pauli rotations out of a 2x2.
[[nodiscard]] const Matrix2x2& ipz();
/// `i * sigma_y`.
[[nodiscard]] const Matrix2x2& ipy();
/// `i * sigma_x`.
[[nodiscard]] const Matrix2x2& ipx();

/// Kronecker-embed a 2x2 on wire ``qubitId`` (identity on the other wire).
[[nodiscard]] Matrix4x4 expandToTwoQubits(const Matrix2x2& singleQubitMatrix,
                                          QubitId qubitId);

/// Reorder a 4x4 two-qubit matrix so its qubits match the canonical
/// `(low, high)` order given the operand-order `qubitIds`. No-op when the
/// operand order already matches.
[[nodiscard]] Matrix4x4
fixTwoQubitMatrixQubitOrder(const Matrix4x4& twoQubitMatrix,
                            const llvm::SmallVector<QubitId, 2>& qubitIds);

/// Construct the 2x2 / 4x4 matrix described by `gate`. Two-qubit gates are
/// returned in the convention matching `expandToTwoQubits` + the gate's own
/// operand order.
[[nodiscard]] Matrix2x2 getSingleQubitMatrix(const Gate& gate);
[[nodiscard]] Matrix4x4 getTwoQubitMatrix(const Gate& gate);

} // namespace mlir::qco::decomposition
