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

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/Utils.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <optional>

namespace mlir::qco {

/// Maximum number of modifier targets supported by @ref
/// composeNTargetBodyMatrix.
inline constexpr std::size_t kMaxModifierTargetQubits = 10;

/**
 * @brief Composes compile-time unitaries in a modifier body on @p numTargets
 * wires.
 *
 * @details Block arguments map to wire indices `0..numTargets-1` (MSB-first,
 * matching @ref Matrix2x2::embedInNqubit). Returns the composed unitary in
 * program order, or `std::nullopt` when the body cannot be composed.
 */
[[nodiscard]] std::optional<DynamicMatrix>
composeBodyMatrix(Block& block, std::size_t numTargets);

/**
 * @brief Check whether two parameter values match.
 *
 * @details
 * Identical SSA values always match. Otherwise, if both are constants, they
 * are compared with @ref utils::TOLERANCE.
 *
 * @param lhs The first parameter value.
 * @param rhs The second parameter value.
 * @return true if the values match.
 */
static bool valuesMatchWithinTolerance(Value lhs, Value rhs) {
  if (lhs == rhs) {
    return true;
  }
  const auto lhsVal = utils::valueToDouble(lhs);
  const auto rhsVal = utils::valueToDouble(rhs);
  return lhsVal && rhsVal && std::abs(*lhsVal - *rhsVal) <= utils::TOLERANCE;
}

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
LogicalResult
removeInversePairOneTargetZeroParameter(OpType op, PatternRewriter& rewriter) {
  // Check if the successor is the inverse operation
  auto nextOp = dyn_cast<InverseOpType>(*op.getOutputQubit(0).user_begin());
  if (!nextOp) {
    return failure();
  }

  // Erase both operations
  rewriter.replaceOp(op, op.getInputQubits());
  rewriter.replaceOp(nextOp, nextOp.getInputQubits());
  return success();
}

/**
 * @brief Remove a pair of inverse two-target, zero-parameter operations.
 *
 * @tparam InverseOpType The type of the inverse operation.
 * @tparam OpType The type of the operation to be checked.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @param symmetric Whether the two-target gate is symmetric (order of the
 * qubits does not matter)
 * @param swappedTargets Whether the successor consumes swapped target wires.
 * @return LogicalResult Success or failure of the removal.
 */
template <typename InverseOpType, typename OpType>
LogicalResult
removeInversePairTwoTargetZeroParameter(OpType op, PatternRewriter& rewriter,
                                        bool symmetric = false,
                                        bool swappedTargets = false) {
  auto output0 = op.getOutputQubit(0);

  // Check if the successor is the inverse operation
  auto nextOp = dyn_cast<InverseOpType>(*output0.user_begin());
  if (!nextOp) {
    return failure();
  }

  // Both qubits have to point to the same successor
  auto nextOp2 = *op.getOutputQubit(1).user_begin();
  if (nextOp2 != nextOp) {
    return failure();
  }

  if (symmetric || (swappedTargets && output0 == nextOp.getInputQubit(1)) ||
      (!swappedTargets && output0 == nextOp.getInputQubit(0))) {
    rewriter.replaceOp(op, op.getInputQubits());
    rewriter.replaceOp(nextOp, nextOp.getInputQubits());
    return success();
  }
  return failure();
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
LogicalResult mergeOneTargetZeroParameter(OpType op,
                                          PatternRewriter& rewriter) {
  // Check if the successor is the same operation
  auto nextOp = dyn_cast<OpType>(*op.getOutputQubit(0).user_begin());
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
LogicalResult mergeOneTargetOneParameter(OpType op, PatternRewriter& rewriter) {
  // Check if the successor is the same operation
  auto nextOp = dyn_cast<OpType>(*op.getOutputQubit(0).user_begin());
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
 * @brief Shared implementation for merging two-target, one-parameter
 * operations.
 *
 * @tparam OpType The type of the operation to be merged.
 * @param op The first operation instance.
 * @param nextOp The successor operation instance.
 * @param rewriter The pattern rewriter.
 * @param symmetric Whether the two-target gate is symmetric (order of the
 * qubits does not matter)
 * @return LogicalResult Success or failure of the merge.
 */
template <typename OpType>
static LogicalResult mergeTwoTargetOneParameterImpl(OpType op, OpType nextOp,
                                                    PatternRewriter& rewriter,
                                                    bool symmetric = false) {

  // Both qubits have to point to the same successor
  auto nextOp2 = *op.getOutputQubit(1).user_begin();
  if (nextOp2 != nextOp) {
    return failure();
  }

  auto output0 = op.getOutputQubit(0);
  if (symmetric || output0 == nextOp.getInputQubit(0)) {
    // Compute and set the new parameter
    auto newParameter = arith::AddFOp::create(
        rewriter, op.getLoc(), op.getOperand(2), nextOp.getOperand(2));
    op->setOperand(2, newParameter.getResult());
    rewriter.replaceOp(nextOp, nextOp.getInputQubits());
    return success();
  }
  return failure();
}

/**
 * @brief Merge two compatible two-target, one-parameter operations.
 *
 * @tparam OpType The type of the operation to be merged.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @param symmetric Whether the two-target gate is symmetric (order of the
 * qubits does not matter)
 * @return LogicalResult Success or failure of the merge.
 */
template <typename OpType>
LogicalResult mergeTwoTargetOneParameter(OpType op, PatternRewriter& rewriter,
                                         bool symmetric = false) {
  // Check if the successor is the same operation
  auto nextOp = dyn_cast<OpType>(*op.getOutputQubit(0).user_begin());
  if (!nextOp) {
    return failure();
  }
  return mergeTwoTargetOneParameterImpl(op, nextOp, rewriter, symmetric);
}

/**
 * @brief Merge consecutive XXPlusYY or XXMinusYY operations.
 *
 * @details
 * Sums `theta` when `beta` matches within tolerance.
 *
 * @tparam OpType The type of the operation to be merged.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the merge.
 */
template <typename OpType>
LogicalResult mergeXXPlusMinusYY(OpType op, PatternRewriter& rewriter) {
  // Check if the successor is the same operation
  auto nextOp = dyn_cast<OpType>(*op.getOutputQubit(0).user_begin());
  if (!nextOp) {
    return failure();
  }

  // Confirm matching beta before summing theta
  if (!valuesMatchWithinTolerance(op.getBeta(), nextOp.getBeta())) {
    return failure();
  }
  return mergeTwoTargetOneParameterImpl(op, nextOp, rewriter, true);
}

} // namespace mlir::qco
