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
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

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
 *       ┌───┐         ┌───┐                            ┌───┐
 *  ──■──┤ X ├    ──■──┤ X ├──■────■──    ──╳────■──    ┤ X ├
 *  ┌─┴─┐└─┬─┘ => ┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐ =>   |  ┌─┴─┐ => └─┬─┘
 *  ┤ X ├──■──    ┤ X ├──■──┤ X ├┤ X ├    ──╳──┤ X ├    ──■──
 *  └───┘         └───┘     └───┘└───┘         └───┘
 */
struct SwapReconstructionAndElisionPattern final : mlir::OpRewritePattern<XOp> {

  explicit SwapReconstructionAndElisionPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  /**
   * @brief If pattern is applicable, perform MLIR rewrite.
   *
   * Steps:
   *   - Find CNOT with exactly one positive control qubit (1st CNOT)
   *   - Check if it has adjacent CNOT with reversed control/target (2nd CNOT)
   *  (- Theoretically place two CNOTs identical to 1st CNOT on other side of
   *     2nd CNOT)
   *   - Remove 1st CNOT and swap output
   */
  mlir::LogicalResult
  matchAndRewrite(XOp op, mlir::PatternRewriter& rewriter) const override {
    if (auto secondCNot = findReverseCNotPattern(op)) {
      rewriter.replaceOpWithNewOp<XOp>(
          op, op.getPosCtrlOutQubits().getType(), op.getOutQubits().getType(),
          mlir::TypeRange{}, mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          op.getPosCtrlInQubits(), op.getInQubits(), mlir::ValueRange{});
      rewriter.replaceOp(*secondCNot, secondCNot->getOperands());
      return mlir::success();
    }

    return mlir::failure();
  }

  /**
   * @brief Checks if two consecutive gates have reversed control and target
   * qubits.
   *
   *         ┌───┐
   * a: ──■──┤ X ├
   *    ┌─┴─┐└─┬─┘
   * b: ┤ X ├──■──
   *    └───┘
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
   * @brief Find a user of the given operation for which isCandidate() is
   * true.
   *
   * @param op Operation for which the users should be scanned for a candidate
   * @param isThirdCNot If true, the candidate should have a subset of
   * controls of op (a surrounding CNOT, 1st/3rd); if false, it should have a
   * superset of controls of op (middle CNOT, 2nd)
   */
  [[nodiscard]] static std::optional<XOp> findReverseCNotPattern(XOp& op) {
    for (auto* user : op->getUsers()) {
      if (auto cnot = llvm::dyn_cast<XOp>(user)) {
        if (isReverseCNotPattern(op, cnot)) {
          llvm::errs() << "YES\n";
          return cnot;
        } else {
          llvm::errs() << "NOPE\n";
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
