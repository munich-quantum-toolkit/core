/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include <mlir/IR/PatternMatch.h>

namespace mlir::flux {

/**
 * @brief Remove a pair of inverse one-target, zero-parameter operations
 *
 * @tparam InverseOpType The type of the inverse operation.
 * @tparam OpType The type of the operation to be checked.
 * @param op The operation instance to be checked.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the removal.
 */
template <typename InverseOpType, typename OpType>
inline mlir::LogicalResult
removeInversePairOneTargetZeroParameter(OpType op,
                                        mlir::PatternRewriter& rewriter) {
  // Check if the predecessor is the inverse operation
  auto prevOp = op.getQubitIn().template getDefiningOp<InverseOpType>();
  if (!prevOp) {
    return failure();
  }

  // Remove both operations
  rewriter.replaceOp(prevOp, prevOp.getQubitIn());
  rewriter.replaceOp(op, op.getQubitIn());

  return success();
}

} // namespace mlir::flux
