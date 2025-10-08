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

#include <cassert>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>

namespace mqt::ir::opt {

/**
 * @brief This pattern attempts to remove uncontrolled SWAP gates by re-ordering
 * qubits.
 */
struct ElidePermutationsPattern final : mlir::OpRewritePattern<SWAPOp> {

  explicit ElidePermutationsPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {
    setHasBoundedRewriteRecursion(true);
  }

  mlir::LogicalResult
  matchAndRewrite(SWAPOp op, mlir::PatternRewriter& rewriter) const override {
    if (op.isControlled()) {
      return mlir::failure();
    }

    const auto inQubits = op.getInQubits();
    assert(inQubits.size() == 2);

    rewriter.replaceOp(op, {inQubits[1], inQubits[0]});

    return mlir::success();
  }
};

/**
 * @brief This pattern attempts to find CNOT patterns which can be replaced by a
 * SWAP gate and directly elides the new permutation.
 *
 * Example (matchTwoCNOTPattern == true):
 *          ┌───┐            ┌───┐                                   ┌───┐
 *  q1 ──■──┤ X ├ q1    ──■──┤ X ├──■────■──    ──╳────■──    q1 ─q0─┤ X ├ q0
 *     ┌─┴─┐└─┬─┘    => ┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐ =>   |  ┌─┴─┐ =>        └─┬─┘
 *  q0 ┤ X ├──■── q0    ┤ X ├──■──┤ X ├┤ X ├    ──╳──┤ X ├    q0 ─q1───■── q1
 *     └───┘            └───┘     └───┘└───┘         └───┘
 *
 * Example (matchTwoCNOTPattern == false)
 *          ┌───┐
 * q1: ──■──┤ X ├──■── q1    q1: ──╳── q1    q1 ─ q0
 *     ┌─┴─┐└─┬─┘┌─┴─┐    =>       |      =>
 * q0: ┤ X ├──■──┤ X ├ q0    q0: ──╳── q0    q0 ─ q1
 *     └───┘     └───┘
 */
template <bool matchTwoCNOTPattern>
struct SwapReconstructionAndElisionPattern final : mlir::OpRewritePattern<XOp> {

  explicit SwapReconstructionAndElisionPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context, benefit()) {
    setHasBoundedRewriteRecursion(true);
  }

  [[nodiscard]] constexpr static mlir::PatternBenefit benefit() {
    return matchTwoCNOTPattern ? 2 : 3;
  }

  /**
   * @brief If pattern is applicable, perform MLIR rewrite.
   *
   * Steps(matchTwoCNOTPattern == true):
   *   1. Find CNOT with exactly one positive control qubit (1st CNOT)
   *   2. Check if it has adjacent CNOT with swapped control/target (2nd CNOT)
   *  (3. Theoretically place two CNOTs identical to 1st CNOT on other side of
   *      2nd CNOT; not actually done because they would be immediately replaced
   *      by SWAP anyway)
   *   4. Replace 1st CNOT with swapped target and control; this swap makes it
   *      equivalent to 2nd CNOT and directly permutates the order of the qubits
   *      to elide the theoretically inserted SWAP
   *   5. Erase 2nd CNOT
   *
   * Steps(matchTwoCNOTPattern == false):
   *   1. Find CNOT with exactly one positive control qubit (1st CNOT)
   *   2. Check if it has adjacent CNOT with swapped control/target (2nd CNOT)
   *   3. Check if 2nd CNOT has another adjacent CNOT with swapped
   *      control/target (3rd CNOT)
   *   4. Erase 1st CNOT and swap output qubits for swap elision
   *   5. Erase 2nd CNOT and 3rd CNOT
   */
  mlir::LogicalResult
  matchAndRewrite(XOp op, mlir::PatternRewriter& rewriter) const override {
    // step 1
    if (op.getPosCtrlInQubits().size() != 1 &&
        op.getNegCtrlInQubits().empty()) {
      return mlir::failure();
    }

    // steps 2
    if (auto secondCNOT = findReverseCNOTPattern(op)) {
      if constexpr (matchTwoCNOTPattern) {
        performTwoCNOTSwapReconstructionAndElision(rewriter, op, *secondCNOT);
        return mlir::success();
      }

      if (auto thirdCNOT = findReverseCNOTPattern(*secondCNOT)) {
        performThreeCNOTSwapReconstructionAndElision(rewriter, op, *secondCNOT,
                                                     *thirdCNOT);
        return mlir::success();
      }
    }

    return mlir::failure();
  }

  /**
   * @brief Perform two-CNOT-swap-reconstruction and elision on two CNOTs for
   * which isReverseCNOTPattern() must return true.
   *
   * @note This is used when matchTwoCNOTPattern is true.
   *
   *          ┌───┐              ┌───┐
   *  q1 ──■──┤ X ├ q1    q1 ─q0─┤ X ├ q0
   *     ┌─┴─┐└─┬─┘    =>        └─┬─┘
   *  q0 ┤ X ├──■── q0    q0 ─q1───■── q1
   *     └───┘
   */
  static void performTwoCNOTSwapReconstructionAndElision(
      mlir::PatternRewriter& rewriter, XOp& firstCNOT, const XOp& secondCNOT) {
    const auto qubitType = QubitType::get(rewriter.getContext());

    // step 3 + 4
    rewriter.replaceOpWithNewOp<XOp>(
        firstCNOT, qubitType, qubitType, mlir::TypeRange{},
        mlir::DenseF64ArrayAttr{}, mlir::DenseBoolArrayAttr{},
        mlir::ValueRange{},
        // swap control and target
        firstCNOT.getPosCtrlInQubits(), firstCNOT.getInQubits(),
        mlir::ValueRange{});
    // step 5
    rewriter.replaceOp(secondCNOT, secondCNOT->getOperands());
  }

  /**
   * @brief Perform swap reconstruction and elision on given CNOT operations
   * which must be equivalent to a swap.
   *
   * This is done by swapping the output of one of the CNOTs and removing all
   * others.
   *
   * @note This is used when matchTwoCNOTPattern is false.
   *
   *          ┌───┐
   * q1: ──■──┤ X ├──■── q1    q1 ─ q0
   *     ┌─┴─┐└─┬─┘┌─┴─┐    =>
   * q0: ┤ X ├──■──┤ X ├ q0    q0 ─ q1
   *     └───┘     └───┘
   */
  static void performThreeCNOTSwapReconstructionAndElision(
      mlir::PatternRewriter& rewriter, const XOp& firstCNOT,
      const XOp& secondCNOT, const XOp& thirdCNOT) {
    // step 4
    rewriter.replaceOp(firstCNOT,
                       {firstCNOT->getOperand(1), firstCNOT->getOperand(0)});
    // step 5
    rewriter.replaceOp(secondCNOT, secondCNOT->getOperands());
    rewriter.replaceOp(thirdCNOT, thirdCNOT->getOperands());
  }

  /**
   * @brief Checks if two consecutive gates have reversed control and target
   * qubits.
   *
   *          ┌───┐
   * q0: ──■──┤ X ├
   *     ┌─┴─┐└─┬─┘
   * q1: ┤ X ├──■──
   *     └───┘
   *
   * @param a The first gate.
   * @param b The second gate which must be one of the users of a.
   * @return True if the gates match the pattern described above, otherwise
   * false.
   */
  [[nodiscard]] static bool isReverseCNOTPattern(XOp& a, XOp& b) {
    return a.getNegCtrlOutQubits().empty() && b.getNegCtrlInQubits().empty() &&
           llvm::equal(a.getOutQubits(), b.getPosCtrlInQubits()) &&
           llvm::equal(b.getInQubits(), a.getPosCtrlOutQubits());
  }

  /**
   * @brief Find a user of the given operation for which isReverseCNOTPattern()
   * is true.
   *
   * @param op Operation for which the users should be scanned for the pattern
   */
  [[nodiscard]] static std::optional<XOp> findReverseCNOTPattern(XOp& op) {
    for (auto* user : op->getUsers()) {
      if (auto cx = llvm::dyn_cast<XOp>(user)) {
        if (isReverseCNOTPattern(op, cx)) {
          return cx;
        }
      }
    }
    return std::nullopt;
  }
};

/**
 * @brief Populates the given pattern set with the `ElidePermutationsPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateSwapReconstructionAndElisionPatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<ElidePermutationsPattern>(patterns.getContext());
  patterns.add<SwapReconstructionAndElisionPattern<true>>(
      patterns.getContext());
  patterns.add<SwapReconstructionAndElisionPattern<false>>(
      patterns.getContext());
}

} // namespace mqt::ir::opt
