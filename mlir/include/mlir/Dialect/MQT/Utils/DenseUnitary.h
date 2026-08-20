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

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>

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
[[nodiscard]] inline LogicalResult
verifyDenseUnitaryMatrix(Operation* operation, ElementsAttr matrixAttr,
                         ValueRange qubits) {
  const auto numQubits = qubits.size();
  if (numQubits == 0U) {
    return operation->emitOpError("requires at least one qubit");
  }
  if (numQubits > MAX_DENSE_UNITARY_QUBITS) {
    return operation->emitOpError()
           << "supports at most " << MAX_DENSE_UNITARY_QUBITS << " qubits";
  }
  llvm::SmallDenseSet<Value, 8> uniqueQubits;
  for (auto qubit : qubits) {
    if (!uniqueQubits.insert(qubit).second) {
      return operation->emitOpError("duplicate qubit operand");
    }
  }
  auto matrix = dyn_cast<DenseElementsAttr>(matrixAttr);
  if (!matrix) {
    return operation->emitOpError("matrix must use dense element storage");
  }
  auto type = dyn_cast<RankedTensorType>(matrix.getType());
  if (!type || type.getRank() != 2 ||
      type.getShape()[0] != type.getShape()[1]) {
    return operation->emitOpError("matrix must be a square rank-two tensor");
  }
  auto complexType = dyn_cast<ComplexType>(type.getElementType());
  if (!complexType || !complexType.getElementType().isF64()) {
    return operation->emitOpError(
        "matrix elements must have type complex<f64>");
  }
  const auto expectedDimension = static_cast<int64_t>(uint64_t{1} << numQubits);
  if (type.getShape()[0] != expectedDimension) {
    return operation->emitOpError()
           << "matrix dimension must be 2^n = " << expectedDimension << " for "
           << numQubits << " qubits";
  }

  llvm::SmallVector<std::complex<double>, 16> values;
  values.reserve(static_cast<size_t>(matrix.size()));
  for (const auto value : matrix.getValues<std::complex<double>>()) {
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
      return operation->emitOpError("matrix entries must be finite");
    }
    values.push_back(value);
  }

  const auto dimension = static_cast<size_t>(expectedDimension);
  for (size_t lhsColumn = 0; lhsColumn < dimension; ++lhsColumn) {
    for (size_t rhsColumn = lhsColumn; rhsColumn < dimension; ++rhsColumn) {
      std::complex<double> innerProduct{};
      for (size_t row = 0; row < dimension; ++row) {
        const auto lhs = values[(row * dimension) + lhsColumn];
        const auto rhs = values[(row * dimension) + rhsColumn];
        innerProduct += std::conj(lhs) * rhs;
      }
      const std::complex<double> expected = lhsColumn == rhsColumn ? 1.0 : 0.0;
      if (!std::isfinite(innerProduct.real()) ||
          !std::isfinite(innerProduct.imag()) ||
          std::abs(innerProduct - expected) > DENSE_UNITARY_TOLERANCE) {
        return operation->emitOpError()
               << "matrix must be unitary within absolute tolerance "
               << DENSE_UNITARY_TOLERANCE;
      }
    }
  }
  return success();
}

/** Return whether a dense square matrix is exactly the identity. */
[[nodiscard]] inline bool isExactIdentityMatrix(ElementsAttr matrixAttr) {
  auto matrix = dyn_cast<DenseElementsAttr>(matrixAttr);
  if (!matrix) {
    return false;
  }
  auto type = dyn_cast<RankedTensorType>(matrix.getType());
  if (!type || type.getRank() != 2 ||
      type.getShape()[0] != type.getShape()[1] || type.getShape()[0] <= 0) {
    return false;
  }
  auto complexType = dyn_cast<ComplexType>(type.getElementType());
  if (!complexType || !complexType.getElementType().isF64()) {
    return false;
  }
  const auto dimension = static_cast<size_t>(type.getShape()[0]);
  size_t index = 0U;
  for (const auto value : matrix.getValues<std::complex<double>>()) {
    const auto row = index / dimension;
    const auto column = index % dimension;
    const std::complex<double> expected = row == column ? 1.0 : 0.0;
    if (value != expected) {
      return false;
    }
    ++index;
  }
  return true;
}

} // namespace mlir::mqt
