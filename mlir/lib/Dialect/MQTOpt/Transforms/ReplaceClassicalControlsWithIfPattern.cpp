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

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <unordered_set>
#include <vector>

namespace mqt::ir::opt {

/**
 * @brief This pattern is responsible for replacing controls after measurements
 * with `if` constructs.
 */
struct ReplaceClassicalControlsWithIfPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit ReplaceClassicalControlsWithIfPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  /**
   * @brief Clones the given Unitary operation without the specified control
   * operand.
   *
   * @param op The Unitary operation to clone.
   * @param operand The control operand to remove.
   * @param positive Whether the control is positive or negative.
   * @param rewriter The pattern rewriter to use for creating the new operation.
   * @return A new operation that is a clone of `op` without the specified
   * control operand.
   */
  static mlir::Operation*
  cloneUnitaryOpWithoutControl(UnitaryInterface op, mlir::Value operand,
                               bool positive, mlir::PatternRewriter& rewriter) {
    rewriter.setInsertionPointAfter(op);
    mlir::SmallVector<mlir::Value> remainingOperands;
    mlir::SmallVector<mlir::Type> remainingTypes;
    for (const auto [index, otherOperand] :
         llvm::enumerate(op.getAllInQubits())) {
      if (operand != otherOperand) {
        remainingOperands.push_back(otherOperand);
      }
    }
    for (auto it = op->getResults().begin() + 1; it != op->getResults().end();
         ++it) {
      remainingTypes.push_back((*it).getType());
    }

    mlir::OperationState state(op.getLoc(), op->getName());
    state.addOperands(remainingOperands);
    state.addTypes(remainingTypes);
    state.addAttributes(op->getAttrs());

    // We need to update the `operandSegmentSizes` and `resultSegmentSizes`
    // attributes to reflect the removed operand.
    mlir::SmallVector<int32_t> inSegSizes(
        op->getAttrOfType<mlir::DenseI32ArrayAttr>("operandSegmentSizes")
            .asArrayRef());
    mlir::SmallVector<int32_t> outSegSizes(
        op->getAttrOfType<mlir::DenseI32ArrayAttr>("resultSegmentSizes")
            .asArrayRef());
    inSegSizes[positive ? 2 : 3] -=
        1; // Adjust the segment size for the removed control
    outSegSizes[positive ? 1 : 2] -=
        1; // Adjust the segment size for the removed control
    state.addAttribute("operandSegmentSizes",
                       rewriter.getDenseI32ArrayAttr(inSegSizes));
    state.addAttribute("resultSegmentSizes",
                       rewriter.getDenseI32ArrayAttr(outSegSizes));
    return rewriter.create(state);
  }

  /**
   * @brief Matches and rewrites a single control of a Unitary operation
   * @param op The Unitary operation to match and rewrite.
   * @param operand The control operand to match.
   * @param positive Whether the control is positive or negative.
   * @param rewriter The pattern rewriter to use for creating the new operation.
   * @return A new UnitaryInterface operation if the control was successfully or
   * `nullptr` if the control could not be matched or rewritten.
   */
  static UnitaryInterface
  matchAndRewriteSingleControl(UnitaryInterface op, mlir::Value operand,
                               bool positive, mlir::PatternRewriter& rewriter) {
    auto* predecessor = operand.getDefiningOp();
    auto predecessorMeasurement = mlir::dyn_cast<MeasureOp>(predecessor);
    if (!predecessorMeasurement) {
      return nullptr; // The operand does not come from a measurement.
    }
    // The control's output is removed, so other operations need to use the
    // previous input now.
    const auto correspondingOutput = op.getCorrespondingOutput(operand);
    rewriter.replaceAllUsesWith(correspondingOutput, operand);

    // We first create a new operation that is the same as before, just with the
    // current control removed
    auto* reducedOp =
        cloneUnitaryOpWithoutControl(op, operand, positive, rewriter);

    // Now we create the `scf.if` operation that uses the measurement result as
    // condition and yields the outcome of the reducedOp.
    const auto outcome = predecessorMeasurement.getOutBit();
    auto ifOp = rewriter.create<mlir::scf::IfOp>(
        op.getLoc(), reducedOp->getResultTypes(), outcome, true);
    auto* satisfiedBlock = positive ? ifOp.thenBlock() : ifOp.elseBlock();
    auto* nonSatisfiedBlock = positive ? ifOp.elseBlock() : ifOp.thenBlock();

    // If the control is satisfied, the reduced operation is executed.
    rewriter.moveOpBefore(reducedOp, satisfiedBlock, satisfiedBlock->begin());
    rewriter.setInsertionPointAfter(reducedOp);
    rewriter.create<mlir::scf::YieldOp>(op.getLoc(), reducedOp->getResults());

    // If the control is not satisfied, we yield the original values of the
    // qubits.
    rewriter.setInsertionPointToStart(nonSatisfiedBlock);
    rewriter.create<mlir::scf::YieldOp>(op.getLoc(), reducedOp->getOperands());

    // All remaining uses of the original op are replaced by the results of the
    // `if` operation. Then, the original operation is erased.
    int offset = 0;
    for (const auto [index, value] : llvm::enumerate(op->getResults())) {
      if (value == correspondingOutput) {
        // Set `offset` to align the values with the results of the `if`
        // operation then skip the operand, as it is already replaced.
        offset = -1;
        continue;
      }
      rewriter.replaceAllUsesWith(
          value, ifOp.getResult(index - op.getParams().size() + offset));
    }
    rewriter.eraseOp(op);

    auto unitaryReplacedOp = mlir::dyn_cast<UnitaryInterface>(reducedOp);
    return unitaryReplacedOp;
  }

  /**
   * @brief Tries to process the control set (positive or negative) of a Unitary
   * operation.
   *
   * Returns a new UnitaryInterface with updated controls once the first control
   * is successfully processed.
   *
   * @param op The Unitary operation to process.
   * @param controls The set of control operands to process.
   * @param isPositive Whether the controls are positive or negative.
   * @param rewriter The pattern rewriter to use for creating the new operation.
   * @return The new UnitaryInterface operation if a control was successfully
   * processed, or `std::nullopt` if no controls were processed.
   */
  std::optional<UnitaryInterface>
  tryProcessControls(UnitaryInterface op, const auto& controls, bool isPositive,
                     mlir::PatternRewriter& rewriter) const {
    for (const auto& operand : controls) {
      if (auto result =
              matchAndRewriteSingleControl(op, operand, isPositive, rewriter)) {
        return result;
      }
    }
    return std::nullopt;
  }

  mlir::LogicalResult
  matchAndRewrite(UnitaryInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    auto foundMatch = false;
    while (true) {
      if (auto result =
              tryProcessControls(op, op.getPosCtrlInQubits(), true, rewriter)) {
        foundMatch = true;
        op = *result; // Update the operation to the new one with the control
                      // removed.
        continue;
      }
      if (auto result = tryProcessControls(op, op.getNegCtrlInQubits(), false,
                                           rewriter)) {
        foundMatch = true;
        op = *result; // Update the operation to the new one with the control
                      // removed.
        continue;
      }

      // If no more controls can be processed, we break out of the loop.
      break;
    }
    return mlir::success(foundMatch);
  }
};

/**
 * @brief Populates the given pattern set with the
 * `ReplaceClassicalControlsWithIfPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateReplaceClassicalControlsWithIfPatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<ReplaceClassicalControlsWithIfPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
