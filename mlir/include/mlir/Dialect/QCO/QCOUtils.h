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

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/Utils.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>

namespace mlir::qco {

/**
 * @brief Check whether two parameter values are equal, using tolerance for
 * constants.
 *
 * @param lhs The first parameter value.
 * @param rhs The second parameter value.
 * @return true if the values match.
 */
static inline bool valuesMatchWithinTolerance(Value lhs, Value rhs) {
  const auto lhsVal = utils::valueToDouble(lhs);
  const auto rhsVal = utils::valueToDouble(rhs);
  if (lhsVal.has_value() && rhsVal.has_value()) {
    return std::abs(*lhsVal - *rhsVal) <= utils::TOLERANCE;
  }
  return lhs == rhs;
}

/**
 * @brief Sum two floating-point parameter values, folding constants when
 * possible.
 *
 * @param rewriter The pattern rewriter.
 * @param loc The location for newly created operations.
 * @param lhs The first summand.
 * @param rhs The second summand.
 * @return Value The sum, as a constant or `arith.addf` result.
 */
static inline Value addFloatParameters(PatternRewriter& rewriter, Location loc,
                                       Value lhs, Value rhs) {
  const auto lhsVal = utils::valueToDouble(lhs);
  const auto rhsVal = utils::valueToDouble(rhs);
  if (lhsVal.has_value() && rhsVal.has_value()) {
    return utils::constantFromScalar(rewriter, loc, *lhsVal + *rhsVal);
  }
  return arith::AddFOp::create(rewriter, loc, lhs, rhs).getResult();
}

/**
 * @brief Find a same-type partner operation reachable through `ctrl` hops on a
 * control wire.
 *
 * @details
 * Returns failure when no partner exists or when the partner is directly
 * adjacent on the same wire (zero `ctrl` hops).
 *
 * @tparam OpType The type of the operation to be merged.
 * @param op The first operation instance.
 * @return FailureOr<OpType> The partner operation, or failure.
 */
template <typename OpType>
static FailureOr<OpType> findPartnerThroughCtrlControlChain(OpType op) {
  Value v = op->getResult(0);
  if (!llvm::isa<QubitType>(v.getType())) {
    return failure();
  }

  unsigned hops = 0;
  while (v.hasOneUse()) {
    Operation* user = *v.getUsers().begin();
    if (auto next = llvm::dyn_cast<OpType>(user);
        next && next->getOperand(0) == v) {
      if (hops == 0) {
        return failure();
      }
      return next;
    }
    auto ctrl = llvm::dyn_cast<CtrlOp>(user);
    if (!ctrl || !llvm::is_contained(ctrl.getControlsIn(), v)) {
      return failure();
    }
    v = ctrl.getOutputForInput(v);
    ++hops;
  }
  return failure();
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
  rewriter.replaceAllUsesWith(nextOp->getResult(0), op.getInputQubit(0));
  rewriter.eraseOp(nextOp);
  rewriter.eraseOp(op);

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
LogicalResult
removeInversePairTwoTargetZeroParameter(OpType op, PatternRewriter& rewriter) {
  // Check if the successor is the inverse operation
  auto nextOp = dyn_cast<InverseOpType>(*op.getOutputQubit(0).user_begin());
  if (!nextOp) {
    return failure();
  }

  // Confirm operations act on the same qubits
  if (op.getOutputQubit(1) != nextOp.getInputQubit(1)) {
    return failure();
  }

  // Erase both operations
  rewriter.replaceAllUsesWith(nextOp->getResults(),
                              {op.getInputQubit(0), op.getInputQubit(1)});
  rewriter.eraseOp(nextOp);
  rewriter.eraseOp(op);

  return success();
}

/**
 * @brief Remove a pair of two-target, zero-parameter operations where
 *        the second operation is the same gate with swapped targets.
 *
 * @tparam OpType The type of the (self-inverse) operation.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the removal.
 */
template <typename OpType>
LogicalResult
removeTwoTargetZeroParameterPairWithSwappedTargets(OpType op,
                                                   PatternRewriter& rewriter) {
  // Check if the successor is the same operation
  auto nextOp = dyn_cast<OpType>(*op.getOutputQubit(0).user_begin());
  if (!nextOp) {
    return failure();
  }

  // Confirm operations act on the same qubits but with swapped targets
  if (op.getOutputQubit(0) != nextOp.getInputQubit(1) ||
      op.getOutputQubit(1) != nextOp.getInputQubit(0)) {
    return failure();
  }

  // Erase both operations
  rewriter.replaceAllUsesWith(nextOp->getResults(),
                              {op.getInputQubit(1), op.getInputQubit(0)});
  rewriter.eraseOp(nextOp);
  rewriter.eraseOp(op);

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
  auto newParameter = addFloatParameters(
      rewriter, op.getLoc(), op.getOperand(1), nextOp.getOperand(1));
  op->setOperand(1, newParameter);

  // Replace the second operation with the result of the first operation
  rewriter.replaceOp(nextOp, op.getResult());
  return success();
}

/**
 * @brief Merge two compatible one-target, two-parameter operations
 *
 * @details
 * Requires matching secondary parameters (e.g., `phi` for `ROp`). The primary
 * parameter (`theta`) is summed.
 *
 * @tparam OpType The type of the operation to be merged.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the merge.
 */
template <typename OpType>
LogicalResult mergeOneTargetTwoParameter(OpType op, PatternRewriter& rewriter) {
  auto nextOp = dyn_cast<OpType>(*op.getOutputQubit(0).user_begin());
  if (!nextOp) {
    return failure();
  }

  if (!valuesMatchWithinTolerance(op.getPhi(), nextOp.getPhi())) {
    return failure();
  }

  auto newTheta = addFloatParameters(rewriter, op.getLoc(), op.getOperand(1),
                                     nextOp.getOperand(1));
  op->setOperand(1, newTheta);
  rewriter.replaceOp(nextOp, op.getResult());
  return success();
}

/**
 * @brief Merge Z-diagonal one-target, zero-parameter gates through a chain of
 * `ctrl` hops on control wires.
 *
 * @details
 * Replaces `op; …; op` on a control wire with `square; …` (e.g. `s; ctrl; s` →
 * `z; ctrl`).
 *
 * @tparam SquareOpType Result of squaring the gate (e.g. `ZOp` for `SOp`).
 * @tparam OpType The Z-diagonal operation to be merged.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the merge.
 */
template <typename SquareOpType, typename OpType>
LogicalResult
mergeOneTargetZeroParameterThroughCtrlControlChain(OpType op,
                                                   PatternRewriter& rewriter) {
  auto partner = findPartnerThroughCtrlControlChain(op);
  if (failed(partner)) {
    return failure();
  }

  rewriter.replaceOpWithNewOp<SquareOpType>(op, op.getInputQubit(0));
  rewriter.replaceOp(*partner, partner->getInputQubit(0));
  return success();
}

/**
 * @brief Merge Z-diagonal one-target, one-parameter gate angles through `ctrl`
 * control wires.
 *
 * @tparam OpType The type of the Z-diagonal operation to be merged.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the merge.
 */
template <typename OpType>
LogicalResult
mergeOneTargetOneParameterThroughCtrlControlChain(OpType op,
                                                  PatternRewriter& rewriter) {
  auto partner = findPartnerThroughCtrlControlChain(op);
  if (failed(partner)) {
    return failure();
  }

  const Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  const Value newTheta =
      addFloatParameters(rewriter, loc, op.getTheta(), partner->getTheta());
  rewriter.modifyOpInPlace(op, [&] { op.getThetaMutable().assign(newTheta); });
  rewriter.replaceOp(*partner, partner->getOperand(0));
  return success();
}

/**
 * @brief Shared implementation for merging two-target, one-parameter
 * operations.
 *
 * @details
 * When @p swappedTargets is false, the successor must consume the same target
 * wires in the same order. When true, the successor may consume swapped
 * targets. The parameter at operand index 2 is summed.
 *
 * @tparam OpType The type of the operation to be merged.
 * @param op The first operation instance.
 * @param rewriter The pattern rewriter.
 * @param swappedTargets Whether the successor consumes swapped target wires.
 * @return LogicalResult Success or failure of the merge.
 */
template <typename OpType>
static LogicalResult mergeTwoTargetOneParameterImpl(OpType op,
                                                    PatternRewriter& rewriter,
                                                    bool swappedTargets) {
  auto nextOp = dyn_cast<OpType>(*op.getOutputQubit(0).user_begin());
  if (!nextOp) {
    return failure();
  }

  if (swappedTargets) {
    if (op.getOutputQubit(0) != nextOp.getInputQubit(1) ||
        op.getOutputQubit(1) != nextOp.getInputQubit(0)) {
      return failure();
    }
  } else if (op.getOutputQubit(1) != nextOp.getInputQubit(1)) {
    return failure();
  }

  auto newParameter = addFloatParameters(
      rewriter, op.getLoc(), op.getOperand(2), nextOp.getOperand(2));
  op->setOperand(2, newParameter);
  if (swappedTargets) {
    rewriter.replaceOp(nextOp, {op.getOutputQubit(1), op.getOutputQubit(0)});
  } else {
    rewriter.replaceOp(nextOp, op.getResults());
  }
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
LogicalResult mergeTwoTargetOneParameter(OpType op, PatternRewriter& rewriter) {
  return mergeTwoTargetOneParameterImpl(op, rewriter, false);
}

/**
 * @brief Merge two compatible two-target, one-parameter operations where the
 *        second operation consumes the outputs with swapped targets.
 *
 * @details
 * This is analogous to mergeTwoTargetOneParameter, but it additionally handles
 * the case where the second operation swaps its target qubits. The new
 * parameter is computed as the sum of the two original parameters.
 *
 * @tparam OpType The type of the operation to be merged.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the merge.
 */
template <typename OpType>
LogicalResult
mergeTwoTargetOneParameterWithSwappedTargets(OpType op,
                                             PatternRewriter& rewriter) {
  return mergeTwoTargetOneParameterImpl(op, rewriter, true);
}

/**
 * @brief Shared implementation for merging two-target, two-parameter
 * operations.
 *
 * @details
 * Requires matching secondary parameters (e.g., `beta` for `XXPlusYYOp` and
 * `XXMinusYYOp`). The primary parameter (`theta`) at operand index 2 is summed.
 * Wire-order requirements are the same as
 * @ref mergeTwoTargetOneParameterImpl.
 *
 * @tparam OpType The type of the operation to be merged.
 * @param op The first operation instance.
 * @param rewriter The pattern rewriter.
 * @param swappedTargets Whether the successor consumes swapped target wires.
 * @return LogicalResult Success or failure of the merge.
 */
template <typename OpType>
static LogicalResult mergeTwoTargetTwoParameterImpl(OpType op,
                                                    PatternRewriter& rewriter,
                                                    bool swappedTargets) {
  auto nextOp = dyn_cast<OpType>(*op.getOutputQubit(0).user_begin());
  if (!nextOp) {
    return failure();
  }

  if (swappedTargets) {
    if (op.getOutputQubit(0) != nextOp.getInputQubit(1) ||
        op.getOutputQubit(1) != nextOp.getInputQubit(0)) {
      return failure();
    }
  } else if (op.getOutputQubit(1) != nextOp.getInputQubit(1)) {
    return failure();
  }

  if (!valuesMatchWithinTolerance(op.getBeta(), nextOp.getBeta())) {
    return failure();
  }

  auto newTheta = addFloatParameters(rewriter, op.getLoc(), op.getOperand(2),
                                     nextOp.getOperand(2));
  op->setOperand(2, newTheta);
  if (swappedTargets) {
    rewriter.replaceOp(nextOp, {op.getOutputQubit(1), op.getOutputQubit(0)});
  } else {
    rewriter.replaceOp(nextOp, op.getResults());
  }
  return success();
}

/**
 * @brief Merge two compatible two-target, two-parameter operations
 *
 * @details
 * Requires matching secondary parameters (e.g., `beta` for `XXPlusYYOp` and
 * `XXMinusYYOp`). The primary parameter (`theta`) is summed.
 *
 * @tparam OpType The type of the operation to be merged.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the merge.
 */
template <typename OpType>
LogicalResult mergeTwoTargetTwoParameter(OpType op, PatternRewriter& rewriter) {
  return mergeTwoTargetTwoParameterImpl(op, rewriter, false);
}

/**
 * @brief Merge two compatible two-target, two-parameter operations
 *
 * @details
 * Requires matching secondary parameters (e.g., `beta` for `XXPlusYYOp` and
 * `XXMinusYYOp`). The primary parameter (`theta`) is summed. The second
 * operation may consume swapped targets.
 *
 * @tparam OpType The type of the operation to be merged.
 * @param op The operation instance.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the merge.
 */
template <typename OpType>
LogicalResult
mergeTwoTargetTwoParameterWithSwappedTargets(OpType op,
                                             PatternRewriter& rewriter) {
  return mergeTwoTargetTwoParameterImpl(op, rewriter, true);
}

} // namespace mlir::qco
