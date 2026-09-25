/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file DDAdapter.h
/// Conversion from canonical QCO matrices to decision diagrams.

#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "mqt/Dialect/QCO/Utils/Matrix.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"

#include <array>
#include <cstddef>
#include <span>
#include <tuple>

namespace mlir::qco {

/// Build a standard gate matrix from a statically sized parameter list.
/// The gate factory checks arity at compile time.
template <typename GateOp, size_t N>
[[nodiscard]] auto
getStandardGateMatrix(const std::array<double, N>& parameters) {
  if constexpr (N == 0) {
    return GateOp::getUnitaryMatrix();
  } else {
    return std::apply(
        [](auto... values) { return GateOp::unitaryMatrix(values...); },
        parameters);
  }
}

/// Embed a QCO unitary matrix into a DD package.
///
/// @param package DD package used to construct the operation.
/// @param matrix Local unitary in QCO operand order.
/// @param numQubits Number of wires in the surrounding state.
/// @param targets Target wires in matrix-operand order.
/// @param controls Sparse DD controls applied to the local matrix.
/// @pre `numQubits <= package.qubits()`. Every target and control is smaller
/// than `numQubits`; targets are unique and disjoint from controls.
/// @return A matrix decision diagram for the embedded operation.
/// Returns an error if the matrix dimension and target count
/// differ or sparse controls accompany a matrix with more than three targets.
[[nodiscard]] auto makeGateDD(dd::Package& package,
                              std::span<const Complex> matrix, size_t numQubits,
                              llvm::ArrayRef<dd::Qubit> targets,
                              const dd::Controls& controls = {})
    -> FailureOr<dd::MatrixDD>;

template <typename Matrix>
  requires requires(const Matrix& matrix) { matrix.entries(); }
[[nodiscard]] auto makeGateDD(dd::Package& package, const Matrix& matrix,
                              const size_t numQubits,
                              const llvm::ArrayRef<dd::Qubit> targets,
                              const dd::Controls& controls = {})
    -> FailureOr<dd::MatrixDD> {
  return makeGateDD(package, matrix.entries(), numQubits, targets, controls);
}

} // namespace mlir::qco
