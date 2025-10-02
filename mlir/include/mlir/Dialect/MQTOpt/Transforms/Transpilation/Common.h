/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"

#include <cstddef>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>
#include <utility>

namespace mqt::ir::opt {
/**
 * @brief 'For' pushes once onto the stack, hence the parent is at depth one.
 */
constexpr std::size_t FOR_PARENT_DEPTH = 1UL;

/**
 * @brief 'If' pushes twice onto the stack, hence the parent is at depth two.
 */
constexpr std::size_t IF_PARENT_DEPTH = 2UL;

/**
 * @brief The datatype for qubit indices. For now, 64bit.
 */
using QubitIndex = std::size_t;

/**
 * @brief Represents a pair of qubit indices.
 */
using QubitIndexPair = std::pair<QubitIndex, QubitIndex>;

/**
 * @brief Represents a SWAP between two qubits.
 */
using Exchange = std::pair<QubitIndex, QubitIndex>;

/**
 * @brief Create a normalized exchange (smaller index first) for consistent
 * hashing/comparison.
 */
[[nodiscard]] inline Exchange makeExchange(const QubitIndex a,
                                           const QubitIndex b) {
  return a < b ? Exchange{a, b} : Exchange{b, a};
}

/**
 * @brief Return true if the function contains "entry_point" in the passthrough
 * attribute.
 */
[[nodiscard]] bool isEntryPoint(mlir::func::FuncOp op);

/**
 * @brief Check if a unitary acts on two qubits.
 * @param u A unitary.
 * @returns True iff the qubit gate acts on two qubits.
 */
[[nodiscard]] bool isTwoQubitGate(UnitaryInterface u);

/**
 * @brief Return input qubit pair for a two-qubit unitary.
 * @param u A two-qubit unitary.
 * @return Pair of SSA values consisting of the first and second in-qubits.
 */
[[nodiscard]] std::pair<mlir::Value, mlir::Value> getIns(UnitaryInterface op);

/**
 * @brief Return output qubit pair for a two-qubit unitary.
 * @param u A two-qubit unitary.
 * @return Pair of SSA values consisting of the first and second out-qubits.
 */
[[nodiscard]] std::pair<mlir::Value, mlir::Value> getOuts(UnitaryInterface op);
} // namespace mqt::ir::opt
