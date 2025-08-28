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

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace mqt::ir::opt {
/**
 * @brief This pattern attempts to find three CNOT gates next to each other
 * which are equivalent to a SWAP. These gates will be removed and replaced by a
 * SWAP operation.
 */
struct SwapReconstructionPattern final : mlir::OpRewritePattern<XOp> {

  explicit SwapReconstructionPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

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
   * @param b The second gate.
   * @return True if the gates match the pattern described above, otherwise
   * false.
   */
  [[nodiscard]] static bool isReverseCNotPattern(XOp& a, XOp& b) {
    // TODO: allow negative ctrl qubits (at least for first CNOT)?
    auto ctrlQubitsA = a.getPosCtrlOutQubits();
    auto ctrlQubitsB = b.getPosCtrlInQubits();
    auto targetQubitsA = a.getOutQubits();
    auto targetQubitsB = b.getInQubits();

    return ctrlQubitsA.size() == 1 && ctrlQubitsB.size() == 1 &&
           ctrlQubitsA == targetQubitsB && targetQubitsA == ctrlQubitsB;
  }

  mlir::LogicalResult
  matchAndRewrite(XOp op, mlir::PatternRewriter& rewriter) const override {
    // operation can only be part of a series of CNOTs that are equivalent to a
    // SWAP if it has exactly one control qubit
    auto ctrlQubits = op.getPosCtrlInQubits();
    if (ctrlQubits.size() != 1) {
      return mlir::failure();
    }

    auto& firstCNot = op;
    if (auto secondCNot = findCandidate(firstCNot)) {
      auto thirdCNot = findCandidate(*secondCNot);
      if (!thirdCNot) {
        // insert self-cancelling CNOT with same control/target as second CNOT
        // before first CNOT
        thirdCNot = insertSelfCancellingCNot(rewriter, *secondCNot);
      }
      replaceWithSwap(rewriter, firstCNot, *secondCNot, *thirdCNot);
      return mlir::success();
    }

    return mlir::failure();
  }

  /**
   * @brief Find a user of the given operation for which isReverseCNotPattern()
   * with the given operation is true.
   */
  static std::optional<XOp> findCandidate(XOp& op) {
    for (auto* user : op->getUsers()) {
      if (auto cnot = llvm::dyn_cast<XOp>(user)) {
        if (isReverseCNotPattern(op, cnot)) {
          return cnot;
        }
      }
    }
    return std::nullopt;
  }

  static XOp duplicateXOp(mlir::PatternRewriter& rewriter, XOp& previousOp,
                          bool swapTargetControl) {
    auto resultType = previousOp.getOutQubits().getType();
    auto posCtrlResultType = previousOp.getPosCtrlOutQubits().getType();
    auto negCtrlResultType = previousOp.getNegCtrlOutQubits().getType();
    auto input = previousOp.getOutQubits();
    auto posCtrlInput = previousOp.getPosCtrlInQubits();
    auto negCtrlInput = previousOp.getNegCtrlInQubits();

    rewriter.setInsertionPointAfter(previousOp);
    if (swapTargetControl) {
      return rewriter.create<XOp>(
          previousOp->getLoc(), resultType, posCtrlResultType,
          negCtrlResultType, mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{}, posCtrlInput, input,
          negCtrlInput);
    }
    return rewriter.create<XOp>(
        previousOp->getLoc(), resultType, posCtrlResultType, negCtrlResultType,
        mlir::DenseF64ArrayAttr{}, mlir::DenseBoolArrayAttr{},
        mlir::ValueRange{}, input, posCtrlInput, negCtrlInput);
  }

  static XOp insertSelfCancellingCNot(mlir::PatternRewriter& rewriter,
                                      XOp& previousOp) {
    auto previousUsers = previousOp->getUsers();

    auto firstOp = duplicateXOp(rewriter, previousOp, true);
    auto secondOp = duplicateXOp(rewriter, firstOp, false);

    rewriter.replaceAllOpUsesWith(previousOp, secondOp);

    // return first inserted operation which will be used in the swap
    // reconstruction
    return firstOp;
  }

  /**
   * @brief Replace the three given XOp by a single SWAPOp.
   */
  static void replaceWithSwap(mlir::PatternRewriter& rewriter, XOp& a, XOp& b,
                              XOp& c) {
    auto inQubits = a.getInQubits();
    assert(!inQubits.empty());
    auto qubitType = inQubits.front().getType();

    auto newSwapLocation = a->getLoc();
    auto newSwapInQubits = a.getAllInQubits();

    auto newSwap = rewriter.create<SWAPOp>(
        newSwapLocation, mlir::TypeRange{qubitType, qubitType},
        mlir::TypeRange{}, mlir::TypeRange{}, mlir::DenseF64ArrayAttr{},
        mlir::DenseBoolArrayAttr{}, mlir::ValueRange{}, newSwapInQubits,
        mlir::ValueRange{}, mlir::ValueRange{});

    auto newSwapOutQubits = newSwap.getOutQubits();
    assert(newSwapOutQubits.size() == 2);

    // replace three operations by single swap; perform swap on output qubits
    rewriter.replaceOp(c, {newSwapOutQubits[1], newSwapOutQubits[0]});
    rewriter.eraseOp(b);
    rewriter.eraseOp(a);
  }
};

/**
 * @brief Populates the given pattern set with the `MergeRotationGatesPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateSwapReconstructionPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<SwapReconstructionPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
