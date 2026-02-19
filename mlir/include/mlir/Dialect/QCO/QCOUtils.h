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
inline mlir::LogicalResult
removeInversePairOneTargetZeroParameter(OpType op,
                                        mlir::PatternRewriter& rewriter) {
  // Check if the predecessor is the inverse operation
  auto prevOp = op.getInputQubit(0).template getDefiningOp<InverseOpType>();
  if (!prevOp) {
    return failure();
  }

  // Trivialize both operations
  rewriter.replaceOp(prevOp, prevOp.getInputQubit(0));
  rewriter.replaceOp(op, op.getInputQubit(0));

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
inline mlir::LogicalResult
removeInversePairTwoTargetZeroParameter(OpType op,
                                        mlir::PatternRewriter& rewriter) {
  // Check if the predecessor is the inverse operation
  auto prevOp = op.getInputQubit(0).template getDefiningOp<InverseOpType>();
  if (!prevOp) {
    return failure();
  }

  // Confirm operations act on same qubits
  if (op.getInputQubit(1) != prevOp.getOutputQubit(1)) {
    return failure();
  }

  // Trivialize both operations
  rewriter.replaceOp(prevOp,
                     {prevOp.getInputQubit(0), prevOp.getInputQubit(1)});
  rewriter.replaceOp(op, {op.getInputQubit(0), op.getInputQubit(1)});

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
inline mlir::LogicalResult
mergeOneTargetZeroParameter(OpType op, mlir::PatternRewriter& rewriter) {
  // Check if the predecessor is the same operation
  auto prevOp = op.getInputQubit(0).template getDefiningOp<OpType>();
  if (!prevOp) {
    return failure();
  }

  // Replace operation with square operation
  auto squareOp =
      rewriter.create<SquareOpType>(op.getLoc(), op.getInputQubit(0));
  rewriter.replaceOp(op, squareOp.getOutputQubit(0));

  // Trivialize predecessor
  rewriter.replaceOp(prevOp, prevOp.getInputQubit(0));

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
inline mlir::LogicalResult
mergeOneTargetOneParameter(OpType op, mlir::PatternRewriter& rewriter) {
  // Check if the predecessor is the same operation
  auto prevOp = op.getInputQubit(0).template getDefiningOp<OpType>();
  if (!prevOp) {
    return failure();
  }

  // Compute and set new parameter
  // For OneTargetOneParameter operations, the parameter has index 1
  auto newParameter = rewriter.create<arith::AddFOp>(
      op.getLoc(), op.getOperand(1), prevOp.getOperand(1));
  op->setOperand(1, newParameter.getResult());

  // Trivialize predecessor
  rewriter.replaceOp(prevOp, prevOp.getInputQubit(0));

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
inline mlir::LogicalResult
mergeTwoTargetOneParameter(OpType op, mlir::PatternRewriter& rewriter) {
  // Check if the predecessor is the same operation
  auto prevOp = op.getInputQubit(0).template getDefiningOp<OpType>();
  if (!prevOp) {
    return failure();
  }

  // Confirm operations act on same qubits
  if (op.getInputQubit(1) != prevOp.getOutputQubit(1)) {
    return failure();
  }

  // Compute and set new parameter
  // For TwoTargetOneParameter operations, the parameter has index 2
  auto newParameter = rewriter.create<arith::AddFOp>(
      op.getLoc(), op.getOperand(2), prevOp.getOperand(2));
  op->setOperand(2, newParameter.getResult());

  // Trivialize predecessor
  rewriter.replaceOp(prevOp,
                     {prevOp.getInputQubit(0), prevOp.getInputQubit(1)});

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
inline mlir::LogicalResult
removeTrivialOneTargetOneParameter(OpType op, mlir::PatternRewriter& rewriter) {
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
inline mlir::LogicalResult
removeTrivialTwoTargetOneParameter(OpType op, mlir::PatternRewriter& rewriter) {
  const auto param = utils::valueToDouble(op.getOperand(2));
  if (!param || std::abs(*param) > utils::TOLERANCE) {
    return failure();
  }

  // Trivialize operation
  rewriter.replaceOp(op, {op.getInputQubit(0), op.getInputQubit(1)});

  return success();
}

} // namespace mlir::qco
