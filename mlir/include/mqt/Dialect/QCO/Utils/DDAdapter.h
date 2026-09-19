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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <optional>
#include <span>
#include <system_error>
#include <utility>
#include <variant>

namespace mlir::qco {

/// Preserve the DD failure category at the LLVM boundary.
[[nodiscard]] inline llvm::Error ddError(const dd::Error& error) {
  auto code = std::errc::invalid_argument;
  switch (error.kind) {
  case dd::Error::Kind::InvalidArgument:
    break;
  case dd::Error::Kind::OutOfRange:
    code = std::errc::result_out_of_range;
    break;
  case dd::Error::Kind::Numerical:
    code = std::errc::state_not_recoverable;
    break;
  case dd::Error::Kind::IO:
    code = std::errc::io_error;
    break;
  }
  return llvm::createStringError(std::make_error_code(code), error.message);
}

template <typename T>
[[nodiscard]] llvm::Expected<T> ddResult(dd::Result<T> result) {
  if (const auto* error = std::get_if<dd::Error>(&result)) {
    return ddError(*error);
  }
  return std::get<T>(std::move(result));
}

[[nodiscard]] inline llvm::Error
ddResult(const std::optional<dd::Error>& error) {
  return error ? ddError(*error) : llvm::Error::success();
}

/// Obtain the canonical matrix for a standard QCO gate operation.
///
/// Fixed gates provide a static `getUnitaryMatrix()` factory. Parameterized
/// gates provide a static `unitaryMatrix(...)` factory with up to three
/// parameters. This helper presents both forms through one runtime-sized
/// parameter view for the QCO interpreter and QIR runtime.
///
/// @tparam GateOp Standard QCO gate operation type.
/// @param parameters Concrete gate parameters in operation order.
/// @return The operation's canonical QCO matrix type.
/// Returns an error if the parameter count does not match the gate.
template <typename GateOp>
[[nodiscard]] auto getStandardGateMatrix(llvm::ArrayRef<double> parameters) {
  if constexpr (requires { GateOp::unitaryMatrix(0., 0., 0.); }) {
    using Matrix = decltype(GateOp::unitaryMatrix(parameters[0], parameters[1],
                                                  parameters[2]));
    if (parameters.size() != 3) {
      return llvm::Expected<Matrix>(llvm::createStringError(
          std::errc::invalid_argument, "Expected three gate parameters"));
    }
    return llvm::Expected<Matrix>(
        GateOp::unitaryMatrix(parameters[0], parameters[1], parameters[2]));
  } else if constexpr (requires { GateOp::unitaryMatrix(0., 0.); }) {
    using Matrix =
        decltype(GateOp::unitaryMatrix(parameters[0], parameters[1]));
    if (parameters.size() != 2) {
      return llvm::Expected<Matrix>(llvm::createStringError(
          std::errc::invalid_argument, "Expected two gate parameters"));
    }
    return llvm::Expected<Matrix>(
        GateOp::unitaryMatrix(parameters[0], parameters[1]));
  } else if constexpr (requires { GateOp::unitaryMatrix(0.); }) {
    using Matrix = decltype(GateOp::unitaryMatrix(parameters[0]));
    if (parameters.size() != 1) {
      return llvm::Expected<Matrix>(llvm::createStringError(
          std::errc::invalid_argument, "Expected one gate parameter"));
    }
    return llvm::Expected<Matrix>(GateOp::unitaryMatrix(parameters[0]));
  } else {
    using Matrix = decltype(GateOp::getUnitaryMatrix());
    if (!parameters.empty()) {
      return llvm::Expected<Matrix>(llvm::createStringError(
          std::errc::invalid_argument, "Expected no gate parameters"));
    }
    return llvm::Expected<Matrix>(GateOp::getUnitaryMatrix());
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
    -> llvm::Expected<dd::MatrixDD>;

template <typename Matrix>
  requires requires(const Matrix& matrix) { matrix.entries(); }
[[nodiscard]] auto makeGateDD(dd::Package& package, const Matrix& matrix,
                              const size_t numQubits,
                              const llvm::ArrayRef<dd::Qubit> targets,
                              const dd::Controls& controls = {})
    -> llvm::Expected<dd::MatrixDD> {
  return makeGateDD(package, matrix.entries(), numQubits, targets, controls);
}

} // namespace mlir::qco
