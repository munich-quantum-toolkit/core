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
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(SWAPOp op, mlir::PatternRewriter& rewriter) const override {
    if (op.isControlled()) {
      return mlir::failure();
    }

    auto inQubits = op.getInQubits();
    assert(inQubits.size() == 2);

    rewriter.replaceAllOpUsesWith(op, {inQubits[1], inQubits[0]});

    rewriter.eraseOp(op);

    return mlir::success();
  }
};

/**
 * @brief This pattern attempts to find CNOT patterns which can be replaced by a
 * SWAP gate and directly elides the new permutation.
 *
 *          ┌───┐            ┌───┐                                   ┌───┐
 *  q1 ──■──┤ X ├ q1    ──■──┤ X ├──■────■──    ──╳────■──    q1 ─q0─┤ X ├ q0
 *     ┌─┴─┐└─┬─┘    => ┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐ =>   |  ┌─┴─┐ =>        └─┬─┘
 *  q0 ┤ X ├──■── q0    ┤ X ├──■──┤ X ├┤ X ├    ──╳──┤ X ├    q0 ─q1───■── q1
 *     └───┘            └───┘     └───┘└───┘         └───┘
 */
struct SwapReconstructionAndElisionPattern final : mlir::OpRewritePattern<XOp> {

  explicit SwapReconstructionAndElisionPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  /**
   * @brief If pattern is applicable, perform MLIR rewrite.
   *
   * Steps:
   *   1. Find CNOT with exactly one positive control qubit (1st CNOT)
   *   2. Check if it has adjacent CNOT with reversed control/target (2nd CNOT)
   *  (3. Theoretically place two CNOTs identical to 1st CNOT on other side of
   *      2nd CNOT; not actually done because they would be immediately replaced
   *      by SWAP anyway)
   *   4. Replace 1st CNOT with swapped target and control; this swap makes it
   *      equivalent to 2nd CNOT and directly permutates the order of the qubits
   *      to elide the theoretically inserted SWAP
   *   5. Erase 2nd CNOT
   */
  mlir::LogicalResult
  matchAndRewrite(XOp op, mlir::PatternRewriter& rewriter) const override {
    // steps 1 + 2
    if (auto secondCNot = findReverseCNotPattern(op)) {
      // step 3 + 4
      rewriter.replaceOpWithNewOp<XOp>(
          op, op.getPosCtrlOutQubits().getType(), op.getOutQubits().getType(),
          op.getNegCtrlOutQubits().getType(), op.getStaticParamsAttr(),
          op.getParamsMaskAttr(), op.getParams(), op.getPosCtrlInQubits(),
          op.getInQubits(), op.getNegCtrlInQubits());
      // step 5
      rewriter.replaceOp(*secondCNot, secondCNot->getOperands());
      return mlir::success();
    }

    return mlir::failure();
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
  [[nodiscard]] static bool isReverseCNotPattern(XOp& a, XOp& b) {
    auto posCtrlQubitsA = a.getPosCtrlOutQubits();
    auto posCtrlQubitsB = b.getPosCtrlInQubits();
    auto negCtrlQubitsA = a.getNegCtrlOutQubits();
    auto negCtrlQubitsB = b.getNegCtrlInQubits();
    auto targetQubitsA = a.getOutQubits();
    auto targetQubitsB = b.getInQubits();

    return negCtrlQubitsA.empty() && negCtrlQubitsB.empty() &&
           posCtrlQubitsA.size() == 1 && posCtrlQubitsB.size() == 1 &&
           targetQubitsA == posCtrlQubitsB && targetQubitsB == posCtrlQubitsA;
  }

  /**
   * @brief Find a user of the given operation for which isReverseCNotPattern()
   * is true.
   *
   * @param op Operation for which the users should be scanned for the pattern
   */
  [[nodiscard]] static std::optional<XOp> findReverseCNotPattern(XOp& op) {
    for (auto* user : op->getUsers()) {
      if (auto cnot = llvm::dyn_cast<XOp>(user)) {
        if (isReverseCNotPattern(op, cnot)) {
          return cnot;
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
  patterns.add<SwapReconstructionAndElisionPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
