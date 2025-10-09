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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationSupport.h"

#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>
#include <string>
#include <unordered_set>

namespace mqt::ir::opt {

static const std::unordered_set<std::string> DIAGONAL_GATES = {
    "i", "z", "s", "sdg", "t", "tdg", "p"};

/**
 * @brief This pattern is responsible for replacing controls after measurements
 * with `if` constructs.
 */
struct ReplaceBasisStateControlsWithIfPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit ReplaceBasisStateControlsWithIfPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  /**
   * @brief Clones the given Unitary operation without the specified control
   * operand.
   *
   * @param op The Unitary operation to clone.
   * @param operand The control operand to remove.
   * @param rewriter The pattern rewriter to use for creating the new operation.
   * @return A new operation that is a clone of `op` without the specified
   * control operand.
   */
  static UnitaryInterface
  cloneUnitaryOpWithoutControl(UnitaryInterface op, mlir::Value operand,
                               mlir::PatternRewriter& rewriter) {
    rewriter.setInsertionPointAfter(op);
    mlir::SmallVector<mlir::Value> remainingOperands;
    mlir::SmallVector<mlir::Type> remainingTypes;

    // Determine whether the replaced operand is a target, positive control or
    // negative control.
    uint8_t groupIndex = 0;
    if (llvm::find(op.getPosCtrlInQubits(), operand) !=
        op.getPosCtrlInQubits().end()) {
      groupIndex = 1;
    } else if (llvm::find(op.getNegCtrlInQubits(), operand) !=
               op.getNegCtrlInQubits().end()) {
      groupIndex = 2;
    }
    if (groupIndex == 0) {
      if (!op.getPosCtrlInQubits().empty()) {
        // By setting the group index to 1, the first control qubit will be
        // adapted as a target qubit when changing the variadic group sizes.
        groupIndex = 1;
      } else {
        // If there are no positive controls, we can also use a negative control
        // by setting the groupIndex to 2. In this case, however, we first need
        // to add an x gate to the corresponding qubit.
        auto targetNegCtrlInput = op.getNegCtrlInQubits().front();
        rewriter.setInsertionPoint(op);
        auto xGate = rewriter.create<XOp>(
            op.getLoc(), targetNegCtrlInput.getType(), mlir::TypeRange{},
            mlir::TypeRange{}, mlir::DenseF64ArrayAttr{},
            mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
            mlir::ValueRange{targetNegCtrlInput}, mlir::ValueRange{},
            mlir::ValueRange{});
        rewriter.replaceUsesWithIf(targetNegCtrlInput,
                                   xGate.getOutQubits().front(),
                                   [&](mlir::OpOperand& operand) {
                                     // We only replace the single use by the
                                     // modified operation.
                                     return operand.getOwner() == op;
                                   });
        groupIndex = 2;
      }
    }

    // First, we add all parameters.
    for (const auto paramOperand : op.getParams()) {
      remainingOperands.emplace_back(paramOperand);
    }
    // Then we add all other operands, except the one that should be removed.
    for (const auto [index, otherOperand] :
         llvm::enumerate(op.getAllInQubits())) {
      if (operand != otherOperand) {
        remainingOperands.emplace_back(otherOperand);
      }
    }
    for (auto it = op->getResults().begin() + 1; it != op->getResults().end();
         ++it) {
      remainingTypes.push_back((*it).getType());
    }

    mlir::OperationState state(op.getLoc(), op->getName());
    state.addOperands(remainingOperands);
    state.addTypes(remainingTypes);

    // We need to update the `operandSegmentSizes` and `resultSegmentSizes`
    // attributes to reflect the removed operand.
    mlir::SmallVector<int32_t> inSegSizes(
        op->getAttrOfType<mlir::DenseI32ArrayAttr>("operandSegmentSizes")
            .asArrayRef());
    mlir::SmallVector<int32_t> outSegSizes(
        op->getAttrOfType<mlir::DenseI32ArrayAttr>("resultSegmentSizes")
            .asArrayRef());
    inSegSizes[groupIndex + 1] -=
        1; // Adjust the segment size for the removed control
    outSegSizes[groupIndex] -=
        1; // Adjust the segment size for the removed control
    state.addAttribute("operandSegmentSizes",
                       rewriter.getDenseI32ArrayAttr(inSegSizes));
    state.addAttribute("resultSegmentSizes",
                       rewriter.getDenseI32ArrayAttr(outSegSizes));
    return mlir::dyn_cast<UnitaryInterface>(rewriter.create(state));
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
    auto reducedOp = cloneUnitaryOpWithoutControl(op, operand, rewriter);

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
    rewriter.create<mlir::scf::YieldOp>(op.getLoc(),
                                        reducedOp.getAllInQubits());

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
      rewriter.replaceAllUsesWith(value, ifOp.getResult(index + offset));
    }
    rewriter.eraseOp(op);

    return reducedOp;
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
      if (DIAGONAL_GATES.count(op->getName().stripDialect().str()) == 1 &&
          op.getAllInQubits().size() > op.getInQubits().size()) {
        // For diagonal gates, targets can also be treated as controls
        // Therefore, we also check if there are more total in qubits than
        // target in qubits (i.e. at least one control exists).
        if (auto result =
                tryProcessControls(op, op.getInQubits(), true, rewriter)) {
          foundMatch = true;
          op = *result; // Update the operation to the new one with the target
                        // removed.
          continue;
        }
      }

      // If no more controls can be processed, we break out of the loop.
      break;
    }
    return mlir::success(foundMatch);
  }
};

/**
 * @brief Populates the given pattern set with the
 * `ReplaceBasisStateControlsWithIfPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateReplaceBasisStateControlsWithIfPatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<ReplaceBasisStateControlsWithIfPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
