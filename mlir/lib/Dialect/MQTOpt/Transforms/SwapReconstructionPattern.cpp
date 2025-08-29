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
 */
template <bool advancedSwapReconstruction>
struct SwapReconstructionPattern final : mlir::OpRewritePattern<XOp> {

  explicit SwapReconstructionPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context, calculateBenefit()) {}

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
      if (auto thirdCNot = findCandidate(*secondCNot)) {
        //         ┌───┐
        // a: ──■──┤ X ├──■── a    a: ──╳── b
        //    ┌─┴─┐└─┬─┘┌─┴─┐   =>      |
        // b: ┤ X ├──■──┤ X ├ b    b: ──╳── a
        //    └───┘     └───┘
        replaceWithSwap(rewriter, *thirdCNot);
        eraseOperation(rewriter, *secondCNot);
        eraseOperation(rewriter, firstCNot);
        return mlir::success();
      }
      if constexpr (advancedSwapReconstruction) {
        //         ┌───┐              ┌───┐                        ┌───┐
        // a: ──■──┤ X ├ a    a: ──■──┤ X ├──■────■── a    a: ──╳──┤ X ├ b
        //    ┌─┴─┐└─┬─┘   =>    ┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐   =>      |  └─┬─┘
        // b: ┤ X ├──■── b    b: ┤ X ├──■──┤ X ├┤ X ├ b    b: ──╳────■── a
        //    └───┘              └───┘     └───┘└───┘
        replaceWithSwap(rewriter, firstCNot);
        // swapTargetControl(rewriter, *secondCNot);
        return mlir::success();
      }
    }

    return mlir::failure();
  }

  static void eraseOperation(mlir::PatternRewriter& rewriter, XOp& op) {
    rewriter.replaceAllOpUsesWith(op, op.getAllInQubits());
    rewriter.eraseOp(op);
  }

  /**
   * @brief Find a user of the given operation for which isReverseCNotPattern()
   * with the given operation is true.
   */
  [[nodiscard]] static std::optional<XOp> findCandidate(XOp& op) {
    for (auto* user : op->getUsers()) {
      if (auto cnot = llvm::dyn_cast<XOp>(user)) {
        if (isReverseCNotPattern(op, cnot)) {
          return cnot;
        }
      }
    }
    return std::nullopt;
  }

  /**
   * @brief Replace the three given XOp by a single SWAPOp.
   */
  static SWAPOp replaceWithSwap(mlir::PatternRewriter& rewriter, XOp& op) {
    auto inQubits = op.getInQubits();
    assert(!inQubits.empty());
    auto qubitType = inQubits.front().getType();

    auto newSwapLocation = op->getLoc();
    auto newSwapInQubits = op.getAllInQubits();

    rewriter.setInsertionPointAfter(op);
    auto newSwap = rewriter.create<SWAPOp>(
        newSwapLocation, mlir::TypeRange{qubitType, qubitType},
        mlir::TypeRange{}, mlir::TypeRange{}, mlir::DenseF64ArrayAttr{},
        mlir::DenseBoolArrayAttr{}, mlir::ValueRange{}, newSwapInQubits,
        mlir::ValueRange{}, mlir::ValueRange{});

    auto newSwapOutQubits = newSwap.getOutQubits();
    assert(newSwapOutQubits.size() == 2);

    // replace operation by swap; perform swap on output qubits
    rewriter.replaceAllOpUsesWith(op, {newSwapOutQubits[1], newSwapOutQubits[0]});
    rewriter.eraseOp(op);
    return newSwap;
  }

  static void swapTargetControl(mlir::PatternRewriter& rewriter, XOp& op) {
    auto location = op->getLoc();

    rewriter.setInsertionPointAfter(op);
    auto newCNot = rewriter.create<XOp>(
        location, op.getPosCtrlOutQubits().getType(),
        op.getOutQubits().getType(),
        op.getNegCtrlOutQubits().getType(), mlir::DenseF64ArrayAttr{},
        mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
        op.getPosCtrlInQubits(), op.getInQubits(), op.getNegCtrlInQubits());

    // rewriter.replaceOp(op, {newCNot.getPosCtrlOutQubits()[0], newCNot.getOutQubits()[0]});
    rewriter.replaceAllOpUsesWith(op, newCNot);
    rewriter.eraseOp(op);
    // assert(op.getInQubits().size() == 1);
    // assert(op.getPosCtrlInQubits().size() == 1);
    // assert(op.getNegCtrlInQubits().size() == 0);
    // rewriter.modifyOpInPlace(op, [&]() {
    //     auto firstOperand = op.getOperand(0);
    //     auto secondOperand = op.getOperand(1);

    //     op.setOperand(0, secondOperand);
    //     op.setOperand(1, firstOperand);
    // });
  }

  static mlir::PatternBenefit calculateBenefit() {
    // prefer simple swap reconstruction
    return advancedSwapReconstruction ? 10 : 100;
  }
};

/**
 * @brief Populates the given pattern set with the simple
 * `SwapReconstructionPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateSwapReconstructionPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<SwapReconstructionPattern<false>>(patterns.getContext());
}

/**
 * @brief Populates the given pattern set with the advanced
 * `SwapReconstructionPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateAdvancedSwapReconstructionPatterns(
    mlir::RewritePatternSet& patterns) {
  // match two different patterns for simple (3 CNOT -> 1 SWAP) and advanced
  // reconstruction (2 CNOT -> 1 SWAP + 1 CNOT) to avoid applying an advanced
  // reconstruction where a simple one would have been possible; an alternative
  // would be to not check both the users and the previous operations to see if
  // it is three CNOTs in the correct configuration
  patterns.add<SwapReconstructionPattern<true>>(patterns.getContext());
}

} // namespace mqt::ir::opt
