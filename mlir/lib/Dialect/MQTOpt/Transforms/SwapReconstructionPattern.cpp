/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <cassert>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetOperations.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>

namespace mqt::ir::opt {
/**
 * @brief This pattern attempts to find three CNOT gates next to each other
 * which are equivalent to a SWAP. These gates will be removed and replaced by a
 * SWAP operation.
 *
 * Examples:
 *       ┌───┐         ┌───┐
 *  ──■──┤ X ├    ──■──┤ X ├──■────■──    ──╳────■──
 *  ┌─┴─┐└─┬─┘ => ┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐ =>   |  ┌─┴─┐
 *  ┤ X ├──■──    ┤ X ├──■──┤ X ├┤ X ├    ──╳──┤ X ├
 *  └───┘         └───┘     └───┘└───┘         └───┘
 *
 *  ──■────■──    ──■────■────■────■──    ──■────■──
 *    |  ┌─┴─┐      |  ┌─┴─┐  |    |        |    |
 *  ──■──┤ X ├ => ──■──┤ X ├──■────■── => ──╳────■──
 *  ┌─┴─┐└─┬─┘    ┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐      |  ┌─┴─┐
 *  ┤ X ├──■──    ┤ X ├──■──┤ X ├┤ X ├    ──╳──┤ X ├
 *  └───┘         └───┘     └───┘└───┘         └───┘
 *
 *  ──□────□──    ──□────□────□────□──    ──□────□──
 *    |  ┌─┴─┐      |  ┌─┴─┐  |    |        |    |
 *  ──■──┤ X ├ => ──■──┤ X ├──■────■── => ──╳────■──
 *  ┌─┴─┐└─┬─┘    ┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐      |  ┌─┴─┐
 *  ┤ X ├──■──    ┤ X ├──■──┤ X ├┤ X ├    ──╳──┤ X ├
 *  └───┘         └───┘     └───┘└───┘         └───┘
 */
template <bool onlyMatchFullSwapPattern, bool matchControlledSwap>
struct SwapReconstructionPattern final : mlir::OpRewritePattern<XOp> {

  explicit SwapReconstructionPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  /**
   *
   */
  [[nodiscard]] static mlir::Value getCNotOutTarget(XOp& op) {
    auto&& outQubits = op.getOutQubits();
    assert(outQubits.size() == 1);
    return outQubits.front();
  }

  [[nodiscard]] static mlir::Value getCNotInTarget(XOp& op) {
    auto&& inQubits = op.getInQubits();
    assert(inQubits.size() == 1);
    return inQubits.front();
  }

  /**
   * @brief Check if operations are suitable for a two CNOT swap reconstruction.
   *
   * @param op One operation to be checked
   * @param nextOp A user of op which should be checked
   * @param nextIsSubset If true, it will be assumed that op is the "main"
   *                     operation which has to have a superset of controls of
   *                     nextOp; if false, the other way around
   */
  [[nodiscard]] static bool isCandidate(XOp& op, XOp& nextOp,
                                        bool nextIsSubset) {
    auto&& opOutTarget = getCNotOutTarget(op);
    auto&& nextInTarget = getCNotInTarget(nextOp);
    auto&& opOutPosCtrlQubits = op.getPosCtrlOutQubits();
    auto&& nextInPosCtrlQubits = nextOp.getPosCtrlInQubits();
    auto&& opOutNegCtrlQubits = op.getNegCtrlOutQubits();
    auto&& nextInNegCtrlQubits = nextOp.getNegCtrlInQubits();

    auto isSubset = [](auto&& set, auto&& subsetCandidate,
                       std::optional<mlir::Value> ignoredMismatch =
                           std::nullopt) -> bool {
      if (subsetCandidate.size() > set.size()) {
        return false;
      }

      for (auto&& element : subsetCandidate) {
        if (!llvm::is_contained(set, element) &&
            (!ignoredMismatch || *ignoredMismatch != element)) {
          return false;
        }
      }
      return true;
    };

    bool targetIsPosCtrl =
        llvm::is_contained(nextInPosCtrlQubits, opOutTarget) &&
        llvm::is_contained(opOutPosCtrlQubits, nextInTarget);

    bool posCtrlMatches =
        nextIsSubset
            ? isSubset(opOutPosCtrlQubits, nextInPosCtrlQubits, opOutTarget)
            : isSubset(nextInPosCtrlQubits, opOutPosCtrlQubits, nextInTarget);
    bool negCtrlMatches =
        nextIsSubset ? isSubset(opOutNegCtrlQubits, nextInNegCtrlQubits)
                     : isSubset(nextInNegCtrlQubits, opOutNegCtrlQubits);

    // TODO: early return for slightly better performance?
    if constexpr (matchControlledSwap) {
      return targetIsPosCtrl && posCtrlMatches && negCtrlMatches;
    } else {
      return targetIsPosCtrl && opOutPosCtrlQubits.size() == 1 &&
             nextInPosCtrlQubits.size() == 1 && opOutNegCtrlQubits.empty() &&
             nextInNegCtrlQubits.empty();
    }
  }

  /**
   * @brief If pattern is applicable, perform MLIR rewrite.
   *
   * Steps (onlyMatchFullSwapPattern == false):
   *   - Find CNOT with at least one positive control qubit (1st CNOT)
   *   - Check if it has adjacent CNOT with superset of control qubits
   *     (2nd CNOT)
   *  (- Theoretically place two CNOTs identical to 1st CNOT on other side of
   *     2nd CNOT)
   *   - Replace 2nd CNOT by SWAP with identical control qubits (if controlled
   *     swaps are enabled), this also cancels out 1st CNOT and one inserted
   *     CNOT); use target of 1st CNOT as second target for the swap
   *   - Move 1st CNOT to other side of SWAP (takes the place of the left-over
   *     inserted CNOT)
   *
   * Steps (onlyMatchFullSwapPattern == true):
   *   - Find CNOT with at least one positive control qubit (1st CNOT)
   *   - Check if it has adjacent CNOT with superset of control qubits
   *     (2nd CNOT)
   *   - Check if 2nd CNOT has adjacent CNOT identical to 1st CNOT (3rd CNOT)
   *   - Replace 2nd CNOT by SWAP with identical control qubits (if controlled
   *     swaps are enabled)
   *   - Erase 1st and 3rd CNOTs
   */
  mlir::LogicalResult
  matchAndRewrite(XOp op, mlir::PatternRewriter& rewriter) const override {
    // check if at least one positive control; rely on other pattern for
    // negative control decomposition (TODO?)
    if (op.getPosCtrlInQubits().empty()) {
      return mlir::failure();
    }

    auto& firstCNot = op;
    if (auto secondCNot = findCandidate(firstCNot, false)) {
      auto secondSwapTarget = getCNotOutTarget(firstCNot);
      if constexpr (onlyMatchFullSwapPattern) {
        // if enabled, check if there is a third CNOT which must be equal
        // to the first one
        if (auto thirdCNot = checkThirdCNot(firstCNot, *secondCNot)) {
          replaceWithSwap(rewriter, *secondCNot, secondSwapTarget);
          eraseOperation(rewriter, firstCNot);
          eraseOperation(rewriter, *thirdCNot);
          return mlir::success();
        }
      } else {
        auto newSwap = replaceWithSwap(rewriter, *secondCNot, secondSwapTarget);
        swapOperationOrder(rewriter, firstCNot, newSwap);
        return mlir::success();
      }
    }

    return mlir::failure();
  }

  /**
   * @brief Remove given operation from circuit.
   */
  static void eraseOperation(mlir::PatternRewriter& rewriter, XOp& op) {
    // "skip" operation by using its input directly as output
    rewriter.replaceAllOpUsesWith(op, op.getAllInQubits());
    // the operation has no more users now and can be deleted
    rewriter.eraseOp(op);
  }

  /**
   * @brief Move any operation by one after another operation.
   *
   * Compared to mlir::PatternRewriter::moveOpAfter(), this function will handle
   * qubits which are used by both operations. However, op must not create new
   * values which are then used by nextOp.
   *
   * @param rewriter Pattern rewriter used to apply changes
   * @param op Operation to be moved
   * @param nextOp Operation after op past which the other operation should be
   * moved; no third operation can be between these operation which uses any of
   * the out qubits of the other operation
   */
  static void swapOperationOrder(mlir::PatternRewriter& rewriter,
                                 mlir::Operation* op, mlir::Operation* nextOp) {
    // collect inputs which must be updated between the two swapped
    llvm::SmallVector<mlir::Value> newOperandsOp(op->getNumOperands());
    llvm::SmallVector<std::pair<std::size_t, mlir::Value>>
        changedOperandsNextOp;
    for (std::size_t i = 0; i < newOperandsOp.size(); ++i) {
      auto&& currentOperand = op->getOperand(i);
      auto&& currentResult = op->getResult(i);
      if (auto newOperandIndex =
              getCorrespondingOutputIndex(nextOp, currentResult)) {
        newOperandsOp[i] = nextOp->getResult(*newOperandIndex);
        changedOperandsNextOp.push_back({*newOperandIndex, currentOperand});
      } else {
        // if operand is not used by nextOp, simply use the current one
        newOperandsOp[i] = currentOperand;
      }
    }

    // update all users of nextOp to now use the result of op instead
    auto&& opResults = op->getResults();
    auto&& nextOpResults = nextOp->getResults();
    llvm::SmallVector<mlir::Value> userUpdates;
    for (auto* user : nextOp->getUsers()) {
      rewriter.modifyOpInPlace(user, [&]() {
        for (auto&& operand : user->getOpOperands()) {
          auto nextOpIt = llvm::find(nextOpResults, operand.get());
          if (nextOpIt != nextOpResults.end()) {
            if (auto nextOpInput = getCorrespondingInput(nextOp, *nextOpIt)) {
              auto opIt = llvm::find(opResults, *nextOpInput);
              if (opIt != opResults.end()) {
                // operand of user which matches a result of nextOp is also a
                // result of op
                user->setOperand(operand.getOperandNumber(), *opIt);
              }
            }
          }
        }
      });
    }

    // update nextOp to use the previous operand of the operation (since it will
    // be moved behind it) as its new operand; only need to update these
    // operands which were operands of the other operation
    rewriter.modifyOpInPlace(nextOp, [&]() {
      for (auto&& [changedIndex, newOperand] : changedOperandsNextOp) {
        nextOp->setOperand(changedIndex, newOperand);
      }
    });
    // if nextOp uses an operand, change the operand to be the corresponding
    // output of op as new operand
    rewriter.modifyOpInPlace(op, [&]() { op->setOperands(newOperandsOp); });
    rewriter.moveOpAfter(op, nextOp);
  }

  /**
   * @brief Find a user of the given operation for which isCandidate() is true.
   *
   * @param op Operation for which the users should be scanned for a candidate
   * @param isThirdCNot If true, the candidate should have a subset of controls
   *                    of op (a surrounding CNOT, 1st/3rd); if false, it should
   *                    have a superset of controls of op (middle CNOT, 2nd)
   */
  [[nodiscard]] static std::optional<XOp> findCandidate(XOp& op,
                                                        bool isThirdCNot) {
    for (auto* user : op->getUsers()) {
      if (auto cnot = llvm::dyn_cast<XOp>(user)) {
        if (isCandidate(op, cnot, isThirdCNot)) {
          return cnot;
        }
      }
    }
    return std::nullopt;
  }

  /**
   * @brief Check if there is a user for a full CNOT pattern equivalent to a
   * SWAP.
   */
  [[nodiscard]] static std::optional<XOp> checkThirdCNot(XOp& firstCNot,
                                                         XOp& secondCNot) {
    // check if gate is equal, ignoring another operation (secondCNot)
    // in-between
    auto equalsThrough = [&](XOp& a, XOp& b, XOp& inbetween) {
      auto&& aOutput = a->getResults();
      auto&& bInput = b->getOperands();

      for (auto&& inbetweenOut : bInput) {
        if (auto inbetweenIn = getCorrespondingInput(inbetween, inbetweenOut)) {
          if (!llvm::is_contained(aOutput, inbetweenIn)) {
            return false;
          }
        } else if (!llvm::is_contained(aOutput, inbetweenOut)) {
          return false;
        }
      }
      return true;
    };
    for (auto* user : secondCNot->getUsers()) {
      if (auto cnot = llvm::dyn_cast<XOp>(user)) {
        if (equalsThrough(firstCNot, cnot, secondCNot)) {
          return cnot;
        }
      }
    }
    return std::nullopt;
  }

  /**
   * @brief Find input qubit for a result qubit value.
   *
   * @note This only works for operations where the indices of operands and
   *       results line up.
   */
  static std::optional<mlir::Value> getCorrespondingInput(mlir::Operation* op,
                                                          mlir::Value out) {
    for (auto&& result : op->getResults()) {
      if (result == out) {
        auto resultIndex = result.getResultNumber();
        return op->getOperand(resultIndex);
      }
    }
    return std::nullopt;
  }

  /**
   * @brief Find result index for an input qubit value.
   *
   * @note This only works for operations where the indices of operands and
   *       results line up.
   */
  static std::optional<std::size_t>
  getCorrespondingOutputIndex(mlir::Operation* op, mlir::Value in) {
    for (auto&& opOperand : op->getOpOperands()) {
      if (opOperand.get() == in) {
        auto operandIndex = opOperand.getOperandNumber();
        return operandIndex;
      }
    }
    return std::nullopt;
  }

  /**
   * @brief Replace the given XOp by a SWAPOp.
   *
   * @param rewriter Pattern rewriter used to apply modifications
   * @param op Operation to be replaced
   * @param secondTargetIn Second target input for new swap operation
   *
   * @note The result qubits will NOT be swapped since that is already done
   *       implicitly by the CNOT pattern.
   */
  static SWAPOp replaceWithSwap(mlir::PatternRewriter& rewriter, XOp& op,
                                const mlir::Value& secondTargetIn) {
    auto firstTarget = getCNotInTarget(op);
    auto qubitType = firstTarget.getType();

    auto newSwapInQubits = {firstTarget, secondTargetIn};
    llvm::SmallVector<mlir::Value> newSwapInPosCtrlQubits;
    for (auto&& posCtrlInQubit : op.getPosCtrlInQubits()) {
      if (posCtrlInQubit != secondTargetIn) {
        newSwapInPosCtrlQubits.push_back(posCtrlInQubit);
      }
    }
    auto newSwapInNegCtrlQubits = op.getNegCtrlInQubits();

    auto newSwapOutType =
        llvm::SmallVector<mlir::Type>{newSwapInQubits.size(), qubitType};
    auto newSwapOutPosCtrlType =
        llvm::SmallVector<mlir::Type>{newSwapInPosCtrlQubits.size(), qubitType};
    auto newSwapOutNegCtrlType = op.getNegCtrlOutQubits().getType();

    rewriter.setInsertionPointAfter(op);

    return rewriter.replaceOpWithNewOp<SWAPOp>(
        op, newSwapOutType, newSwapOutPosCtrlType, newSwapOutNegCtrlType,
        mlir::DenseF64ArrayAttr{}, mlir::DenseBoolArrayAttr{},
        mlir::ValueRange{}, newSwapInQubits, newSwapInPosCtrlQubits,
        newSwapInNegCtrlQubits);
  }
};

/**
 * @brief Populates the given pattern set with the
 * `SwapReconstructionPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateSwapReconstructionPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<SwapReconstructionPattern<false, false>>(patterns.getContext());
  // only match controlled swap on full three CNOT pattern since this cannot be
  // cancelled out by an elide permutations optimization
  patterns.add<SwapReconstructionPattern<true, true>>(patterns.getContext());
}

} // namespace mqt::ir::opt
