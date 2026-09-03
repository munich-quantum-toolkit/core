/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file DDAdapter.h
 * @brief Conversion from canonical QCO matrices to decision diagrams.
 */

#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/ArrayRef.h>

#include <cstddef>
#include <stdexcept>

namespace mlir::qco {

/**
 * @brief Obtain the canonical matrix for a standard QCO gate operation.
 *
 * Fixed gates provide a static `getUnitaryMatrix()` factory. Parameterized
 * gates provide a static `unitaryMatrix(...)` factory with up to three
 * parameters. This helper presents both forms through one runtime-sized
 * parameter view for the QCO interpreter and QIR runtime.
 *
 * @tparam GateOp Standard QCO gate operation type.
 * @param parameters Concrete gate parameters in operation order.
 * @return The operation's canonical QCO matrix.
 * @throws std::invalid_argument If the parameter count does not match the gate.
 */
template <typename GateOp>
[[nodiscard]] auto getStandardGateMatrix(llvm::ArrayRef<double> parameters)
    -> DynamicMatrix {
  if constexpr (requires { GateOp::unitaryMatrix(0., 0., 0.); }) {
    if (parameters.size() != 3) {
      throw std::invalid_argument("Expected three gate parameters");
    }
    return DynamicMatrix{
        GateOp::unitaryMatrix(parameters[0], parameters[1], parameters[2])};
  } else if constexpr (requires { GateOp::unitaryMatrix(0., 0.); }) {
    if (parameters.size() != 2) {
      throw std::invalid_argument("Expected two gate parameters");
    }
    return DynamicMatrix{GateOp::unitaryMatrix(parameters[0], parameters[1])};
  } else if constexpr (requires { GateOp::unitaryMatrix(0.); }) {
    if (parameters.size() != 1) {
      throw std::invalid_argument("Expected one gate parameter");
    }
    return DynamicMatrix{GateOp::unitaryMatrix(parameters[0])};
  } else {
    if (!parameters.empty()) {
      throw std::invalid_argument("Expected no gate parameters");
    }
    return DynamicMatrix{GateOp::getUnitaryMatrix()};
  }
}

/**
 * @brief Embed a QCO unitary matrix into a DD package.
 *
 * @param package DD package used to construct the operation.
 * @param matrix Local unitary in QCO operand order.
 * @param numQubits Number of wires in the surrounding state.
 * @param targets Target wires in matrix-operand order.
 * @param controls Sparse DD controls applied to the local matrix.
 * @return A matrix decision diagram for the embedded operation.
 * @throws std::invalid_argument If the matrix dimension and target count differ
 *         or sparse controls accompany a matrix with more than three targets.
 */
[[nodiscard]] auto
makeGateDD(dd::Package& package, const DynamicMatrix& matrix, size_t numQubits,
           llvm::ArrayRef<dd::Qubit> targets, const dd::Controls& controls = {})
    -> dd::MatrixDD;

} // namespace mlir::qco
