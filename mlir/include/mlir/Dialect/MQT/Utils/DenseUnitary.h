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

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>

namespace mlir::mqt {

/**
 * Maximum absolute entry-wise deviation of U^dagger U from the identity.
 *
 * This tolerance accounts for binary64 accumulation error while remaining
 * substantially below the precision at which dense input matrices are
 * normally specified.
 */
inline constexpr double DENSE_UNITARY_TOLERANCE = 1e-10;

/** Maximum matrix arity accepted by the deterministic unitarity verifier. */
inline constexpr size_t MAX_DENSE_UNITARY_QUBITS = 8;

/** Verify the common dense-matrix contract of QC and QCO unitary operations. */
[[nodiscard]] LogicalResult verifyDenseUnitaryMatrix(Operation* operation,
                                                     ElementsAttr matrixAttr,
                                                     ValueRange qubits);

/** Return whether a dense square matrix is exactly the identity. */
[[nodiscard]] bool isExactIdentityMatrix(ElementsAttr matrixAttr);

} // namespace mlir::mqt
