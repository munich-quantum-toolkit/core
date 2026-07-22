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

#include "dd/Package_fwd.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Support/LogicalResult.h>

namespace mlir::qco {

/**
 * @brief Sequentially build a matrix DD for a static unitary QCO `func.func`.
 *
 * @details Walks the entry block of @p func, maps `qco.static` SSA values to
 * wire indices (or, if none are present, qubit-typed function arguments as
 * wires `0..n-1`), and applies unitary operations via decision-diagram
 * multiplication.
 *
 * Supported programs:
 * - Standard single- and two-qubit gates with compile-time constant parameters
 *   (sparse DD path)
 * - `ctrl` with a sole standard-gate body (same sparse path)
 * - Other `UnitaryOpInterface` ops with a compile-time known matrix (`inv`,
 *   compound `ctrl`, ...), including `gphase` and `barrier`
 * - Skips: `static`, `sink`, `func.return`, `arith.constant`
 *
 * Known one- and two-qubit matrices are constructed directly as DD gates. The
 * dense matrix fallback accepts full-width unitaries on wires `0..n-1` and
 * rewrites the QCO/MSB-first basis into the DD package's LSB-first indexing.
 * Only this full-width fallback is limited to 12 qubits (dense `2^n × 2^n`
 * storage). Measurements, resets, symbolic parameters, and control-flow ops
 * are not supported.
 *
 * @param func The QCO function to construct the functionality for
 * @param dd The DD package to use (must hold at least the function's qubits)
 * @return The matrix DD on success, or failure for unsupported programs
 */
FailureOr<dd::MatrixDD> buildFunctionality(func::FuncOp func, dd::Package& dd);

/**
 * @brief Simulate a static unitary QCO `func.func` on a given input state.
 *
 * @details Same supported op set and limitations as @ref buildFunctionality.
 * Mirrors @ref dd::simulate: sequentially applies unitaries to @p in via
 * decision-diagram multiplication. Only purely quantum, measurement-free
 * programs are supported.
 *
 * @param func The QCO function to simulate
 * @param in The input state, represented as a vector DD
 * @param dd The DD package to use (must hold at least the function's qubits)
 * @return The output statevector DD on success, or failure for unsupported
 *         programs
 */
FailureOr<dd::VectorDD> simulate(func::FuncOp func, const dd::VectorDD& in,
                                 dd::Package& dd);

} // namespace mlir::qco
