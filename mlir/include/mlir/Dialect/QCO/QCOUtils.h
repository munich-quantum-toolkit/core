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

#include "mlir/Dialect/Utils/Utils.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::qco {

/**
 * @brief Remove a pair of inverse one-target, zero-parameter operations
 *
 * @tparam InverseOpType The type of the inverse operation.
 * @tparam OpType The type of the operation to be checked.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the removal.
 */
template <typename InverseOpType, typename OpType>
mlir::LogicalResult
removeInversePairOneTargetZeroParameter(OpType op, PatternRewriter& rewriter) {
  // Check if the successor is the inverse operation
  auto nextOp =
      llvm::dyn_cast<InverseOpType>(*op.getOutputQubit(0).user_begin());
  if (!nextOp) {
    return failure();
  }

  // Unlink both operations
  rewriter.replaceAllUsesWith(nextOp->getResult(0), op.getInputQubit(0));

  return success();
}

/**
 * @brief Remove a pair of inverse two-target, zero-parameter operations
 *
 * @tparam InverseOpType The type of the inverse operation.
 * @tparam OpType The type of the operation to be checked.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the removal.
 */
template <typename InverseOpType, typename OpType>
mlir::LogicalResult
removeInversePairTwoTargetZeroParameter(OpType op, PatternRewriter& rewriter) {
  // Check if the successor is the inverse operation
  auto nextOp =
      llvm::dyn_cast<InverseOpType>(*op.getOutputQubit(0).user_begin());
  if (!nextOp) {
    return failure();
  }

  // Confirm operations act on the same qubits
  if (op.getOutputQubit(1) != nextOp.getInputQubit(1)) {
    return failure();
  }

  // Unlink both operations
  rewriter.replaceAllUsesWith(nextOp->getResults(), op.getOperands());

  return success();
}

/**
 * @brief Merge two compatible one-target, zero-parameter operations
 *
 * @details
 * The two operations are replaced by a single operation corresponding to their
 * square.
 *
 * @tparam SquareOpType The type of the square operation to be created.
 * @tparam OpType The type of the operation to be merged.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the merge.
 */
template <typename SquareOpType, typename OpType>
mlir::LogicalResult mergeOneTargetZeroParameter(OpType op,
                                                PatternRewriter& rewriter) {
  // Check if the successor is the same operation
  auto nextOp = llvm::dyn_cast<OpType>(*op.getOutputQubit(0).user_begin());
  if (!nextOp) {
    return failure();
  }

  // Replace the first operation with the square operation
  auto newOp =
      rewriter.replaceOpWithNewOp<SquareOpType>(op, op.getInputQubit(0));

  // Replace the second operation with the result of the square operation
  rewriter.replaceOp(nextOp, newOp.getResult());

  return success();
}

/**
 * @brief Merge two compatible one-target, one-parameter operations
 *
 * @details
 * The new parameter is computed as the sum of the two original parameters.
 *
 * @tparam OpType The type of the operation to be merged.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the merge.
 */
template <typename OpType>
mlir::LogicalResult mergeOneTargetOneParameter(OpType op,
                                               PatternRewriter& rewriter) {
  // Check if the successor is the same operation
  auto nextOp = llvm::dyn_cast<OpType>(*op.getOutputQubit(0).user_begin());
  if (!nextOp) {
    return failure();
  }

  // Compute and set the new parameter
  auto newParameter = arith::AddFOp::create(
      rewriter, op.getLoc(), op.getOperand(1), nextOp.getOperand(1));
  op->setOperand(1, newParameter.getResult());

  // Replace the second operation with the result of the first operation
  rewriter.replaceOp(nextOp, op.getResult());

  return success();
}

/**
 * @brief Merge two compatible two-target, one-parameter operations
 *
 * @details
 * The new parameter is computed as the sum of the two original parameters.
 *
 * @tparam OpType The type of the operation to be merged.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the merge.
 */
template <typename OpType>
mlir::LogicalResult mergeTwoTargetOneParameter(OpType op,
                                               PatternRewriter& rewriter) {
  // Check if the successor is the same operation
  auto nextOp = llvm::dyn_cast<OpType>(*op.getOutputQubit(0).user_begin());
  if (!nextOp) {
    return failure();
  }

  // Confirm operations act on the same qubits
  if (op.getOutputQubit(1) != nextOp.getInputQubit(1)) {
    return failure();
  }

  // Compute and set the new parameter
  auto newParameter = arith::AddFOp::create(
      rewriter, op.getLoc(), op.getOperand(2), nextOp.getOperand(2));
  op->setOperand(2, newParameter.getResult());

  // Replace the second operation with the result of the first operation
  rewriter.replaceOp(nextOp, op.getResults());
  return success();
}

/**
 * @brief Remove a trivial one-target, one-parameter operation
 *
 * @tparam OpType The type of the operation to be checked.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the removal.
 */
template <typename OpType>
mlir::LogicalResult
removeTrivialOneTargetOneParameter(OpType op, PatternRewriter& rewriter) {
  const auto param = utils::valueToDouble(op.getOperand(1));
  if (!param || std::abs(*param) > utils::TOLERANCE) {
    return failure();
  }

  // Trivialize operation
  rewriter.replaceOp(op, op.getInputQubit(0));

  return success();
}

/**
 * @brief Remove a trivial two-target, one-parameter operation
 *
 * @tparam OpType The type of the operation to be checked.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the removal.
 */
template <typename OpType>
mlir::LogicalResult
removeTrivialTwoTargetOneParameter(OpType op, PatternRewriter& rewriter) {
  const auto param = utils::valueToDouble(op.getOperand(2));
  if (!param || std::abs(*param) > utils::TOLERANCE) {
    return failure();
  }

  // Trivialize operation
  rewriter.replaceOp(op, op.getInputQubits());

  return success();
}

} // namespace mlir::qco
