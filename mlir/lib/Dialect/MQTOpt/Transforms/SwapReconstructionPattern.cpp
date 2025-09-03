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
 *  ──■──┤ X ├    ──■──┤ X ├──■────■──    ──╳────■──
 *  ┌─┴─┐└─┬─┘ => ┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐ =>   |  ┌─┴─┐
 *  ┤ X ├──■──    ┤ X ├──■──┤ X ├┤ X ├    ──╳──┤ X ├
 *  └───┘         └───┘     └───┘└───┘         └───┘
 *
 *  ──□────□──    ──□────□────□────□──    ──□────□──
 *    |  ┌─┴─┐      |  ┌─┴─┐  |    |        |    |
 *  ──■──┤ X ├    ──■──┤ X ├──■────■──    ──╳────■──
 *  ┌─┴─┐└─┬─┘ => ┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐ =>   |  ┌─┴─┐
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

  [[nodiscard]] static bool isCandidate(XOp& op, XOp& nextOp) {
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

      for (auto&& element : set) {
        if (!llvm::is_contained(subsetCandidate, element) &&
            (!ignoredMismatch || *ignoredMismatch == element)) {
          return false;
        }
      }
      return true;
    };

    // llvm::SmallSetVector<mlir::Value, 4> posCtrlDiff{
    //     nextInPosCtrlQubits.begin(), nextInPosCtrlQubits.end()};
    // posCtrlDiff.set_subtract(opOutPosCtrlQubits);
    // bool posCtrlMatches = posCtrlDiff.size() == 1 && posCtrlDiff.front() ==
    // opOutTarget;

    // bool negCtrlMatches =
    //     llvm::set_is_subset(opOutNegCtrlQubits, nextInNegCtrlQubits);

    bool targetIsPosCtrl =
        llvm::is_contained(nextInPosCtrlQubits, opOutTarget) &&
        llvm::is_contained(opOutPosCtrlQubits, nextInTarget);

    bool posCtrlMatches =
        isSubset(opOutPosCtrlQubits, nextInPosCtrlQubits, opOutTarget);
    bool negCtrlMatches = isSubset(opOutNegCtrlQubits, nextInNegCtrlQubits);

    // TODO: early return possible for better performance?
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
   * Steps:
   *   - Find CNOT with at least one control qubit (1st CNOT)
   *   - Check if it has adjacent CNOT with subset of control qubits (2nd CNOT)
   *  (- Theoretically place two CNOTs identical to 2nd CNOT on other side of
   *     1st CNOT)
   *   - Replace 1st CNOT by SWAP with identical control qubits (also cancels
   *     out 2nd CNOT and one inserted CNOT); use target of 2nd CNOT as second
   *     target for the swap
   *   - Move 2nd CNOT to other side of SWAP (takes the place of the left-over
   *     inserted CNOT)
   */
  mlir::LogicalResult
  matchAndRewrite(XOp op, mlir::PatternRewriter& rewriter) const override {
    // check if at least one positive control; rely on other pattern for
    // negative control decomposition
    if (op.getPosCtrlInQubits().empty()) {
      return mlir::failure();
    }

    auto& firstCNot = op;
    if (auto secondCNot = findCandidate(firstCNot)) {
      auto secondSwapTargetOut = getCNotInTarget(*secondCNot);
      auto newSwap = replaceWithSwap(rewriter, firstCNot, secondSwapTargetOut);
      // rewriter.moveOpAfter(*secondCNot, newSwap);
      return mlir::success();
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
   * @brief Find a user of the given operation for which isReverseCNotPattern()
   * with the given operation is true.
   */
  [[nodiscard]] static std::optional<XOp> findCandidate(XOp& op) {
    for (auto* user : op->getUsers()) {
      if (auto cnot = llvm::dyn_cast<XOp>(user)) {
        if (isCandidate(op, cnot)) {
          return cnot;
        }
      }
    }
    return std::nullopt;
  }

  static mlir::Value getCorrespondingInput(mlir::Operation* op,
                                           mlir::Value out) {
    for (auto&& result : op->getResults()) {
      if (result == out) {
        auto resultIndex = result.getResultNumber();
        return op->getOperand(resultIndex);
      }
    }
  }

  /**
   * @brief Replace the given XOp by a single SWAPOp.
   *
   * @note The result qubits will NOT be swapped since that is already done
   * implicitly by the CNOT pattern.
   */
  static SWAPOp replaceWithSwap(mlir::PatternRewriter& rewriter, XOp& op,
                                const mlir::Value& secondTargetOut) {
    auto firstTarget = getCNotInTarget(op);
    auto secondTargetIn = getCorrespondingInput(op, secondTargetOut);
    auto qubitType = firstTarget.getType();

    auto newSwapLocation = op->getLoc();
    auto newSwapInQubits = {firstTarget,
                            secondTargetIn};
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
    auto newSwap = rewriter.create<SWAPOp>(
        newSwapLocation, newSwapOutType, newSwapOutPosCtrlType,
        newSwapOutNegCtrlType, mlir::DenseF64ArrayAttr{},
        mlir::DenseBoolArrayAttr{}, mlir::ValueRange{}, newSwapInQubits,
        newSwapInPosCtrlQubits, newSwapInNegCtrlQubits);

    rewriter.replaceOp(op, newSwap);
    return newSwap;
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
