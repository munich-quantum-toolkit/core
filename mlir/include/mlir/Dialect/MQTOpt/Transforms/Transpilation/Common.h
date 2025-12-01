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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/QubitIndex.h"

#include <cstddef>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <utility>

namespace mqt::ir::opt {

/**
 * @brief A pair of SSA Values.
 */
using ValuePair = std::pair<mlir::Value, mlir::Value>;

/**
 * @brief Represents a pair of qubit indices.
 */
using QubitIndexPair = std::pair<QubitIndex, QubitIndex>;

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
[[nodiscard]] bool isTwoQubitGate(UnitaryInterface op);

/**
 * @brief Return input qubit pair for a two-qubit unitary.
 * @param op A two-qubit unitary.
 * @return Pair of SSA values consisting of the first and second in-qubits.
 */
[[nodiscard]] ValuePair getIns(UnitaryInterface op);

/**
 * @brief Return output qubit pair for a two-qubit unitary.
 * @param op A two-qubit unitary.
 * @return Pair of SSA values consisting of the first and second out-qubits.
 */
[[nodiscard]] ValuePair getOuts(UnitaryInterface op);

/**
 * @brief Return the first user of a value in a given region.
 * @param v The value.
 * @param region The targeted region.
 * @return A pointer to the user, or nullptr if non exists.
 */
[[nodiscard]] mlir::Operation* getUserInRegion(mlir::Value v,
                                               mlir::Region* region);

/**
 * @brief Create and return SWAPOp for two qubits.
 *
 * Expects the rewriter to be set to the correct position.
 *
 * @param location The Location to attach to the created op.
 * @param in0 First input qubit SSA value.
 * @param in1 Second input qubit SSA value.
 * @param rewriter A PatternRewriter.
 * @return The created SWAPOp.
 */
[[nodiscard]] SWAPOp createSwap(mlir::Location location, mlir::Value in0,
                                mlir::Value in1,
                                mlir::PatternRewriter& rewriter);

/**
 * @brief Replace all uses of a value within a region and its nested regions,
 * except for a specific operation.
 *
 * @param oldValue The value to replace.
 * @param newValue The new value to use.
 * @param region The region in which to perform replacements.
 * @param exceptOp Operation to exclude from replacements.
 * @param rewriter The pattern rewriter.
 */
void replaceAllUsesInRegionAndChildrenExcept(mlir::Value oldValue,
                                             mlir::Value newValue,
                                             mlir::Region* region,
                                             mlir::Operation* exceptOp,
                                             mlir::PatternRewriter& rewriter);

/**
 * @brief Given a layout, validate if the two-qubit unitary op is executable on
 * the targeted architecture.
 *
 * @param op The two-qubit unitary.
 * @param layout The current layout.
 * @param arch The targeted architecture.
 */
[[nodiscard]] bool isExecutable(UnitaryInterface op, const Layout& layout,
                                const Architecture& arch);

/**
 * @brief Insert SWAP ops at the rewriter's insertion point.
 *
 * @param loc The location of the inserted SWAP ops.
 * @param swaps The hardware indices of the SWAPs.
 * @param layout The current layout.
 * @param rewriter The pattern rewriter.
 */
void insertSWAPs(mlir::Location loc, mlir::ArrayRef<QubitIndexPair> swaps,
                 Layout& layout, mlir::PatternRewriter& rewriter);
} // namespace mqt::ir::opt
