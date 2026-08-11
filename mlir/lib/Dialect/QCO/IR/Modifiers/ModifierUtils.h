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

#include <llvm/ADT/STLFunctionalExtras.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>

namespace mlir {

class Block;
class Operation;

namespace qco::detail {

/**
 * @brief Verify the operations and SSA captures in a QCO modifier body.
 */
[[nodiscard]] LogicalResult verifyModifierBody(Operation* modifierOp,
                                               Block& body);

/**
 * @brief Return the positions of the qubits that the body of a modifier uses.
 *
 * @details Qubits are threaded through the body, so a qubit that the body only
 * yields back is not acted upon and can be dropped from the modifier.
 */
[[nodiscard]] SmallVector<size_t> getUsedQubitIndices(Block& body);

/**
 * @brief Rebuild a modifier with only the qubits used by its body.
 *
 * @param modifierOp The modifier to replace.
 * @param body The modifier body.
 * @param qubits The modifier qubits corresponding to the body arguments.
 * @param rebuild A callback that creates and returns the narrowed modifier.
 * @param rewriter The rewriter used to replace the modifier.
 */
[[nodiscard]] LogicalResult
dropUnusedQubits(Operation* modifierOp, Block& body, ValueRange qubits,
                 function_ref<Operation*(ValueRange, ArrayRef<size_t>)> rebuild,
                 RewriterBase& rewriter);

/**
 * @brief Inline @p body into the modifier currently being built, dropping the
 * qubits it does not use, and return the qubits it yields for the others.
 *
 * @details The block arguments of unused qubits are only yielded, and those
 * yields are dropped, so they are replaced with the qubits of the original
 * modifier.
 */
[[nodiscard]] SmallVector<Value>
inlineNarrowedBody(Block& body, ValueRange qubits, ArrayRef<size_t> used,
                   ValueRange args, RewriterBase& rewriter);

/** @brief Replace used positions in @p inputs with @p narrowedResults. */
[[nodiscard]] SmallVector<Value>
restoreUnusedQubits(ValueRange inputs, ArrayRef<size_t> used,
                    ValueRange narrowedResults);

} // namespace qco::detail

} // namespace mlir
