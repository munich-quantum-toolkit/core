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

#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

namespace mlir {

class Block;
class Operation;

namespace qc::detail {

/**
 * @brief Verify the operations and SSA captures in a QC modifier body.
 */
[[nodiscard]] LogicalResult verifyModifierBody(Operation* modifierOp,
                                               Block& body);

/**
 * @brief Inline @p body into the modifier currently being built, dropping the
 * qubits that the body does not use.
 *
 * @details The block arguments of unused qubits have no uses, so they are
 * replaced with the corresponding qubit of the original modifier.
 */
void inlineNarrowedBody(Block& body, ValueRange qubits, ValueRange args,
                        RewriterBase& rewriter);

} // namespace qc::detail

} // namespace mlir
