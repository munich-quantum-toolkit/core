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

  /**
   * @brief Construct a pattern that replaces basis-state controls with
   * conditional `scf::If` regions.
   *
   * Binds the pattern to the provided MLIR context so it can be added to a
   * RewritePatternSet and used by the pattern rewriter.
   */
  explicit ReplaceBasisStateControlsWithIfPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  /**
   * @brief Produce a clone of a Unitary operation with a specified control
   * removed.
   *
   * Creates a new Unitary operation identical to `op` but without `operand` as
   * a control: parameters are preserved, the specified in-qubit is omitted, and
   * the corresponding result is removed. If `operand` is a negative control and
   * there are no positive controls, an X gate is inserted on that qubit and the
   * op's uses are rewired so the new X gate output is used by the cloned op.
   * The returned operation's operand and result segment size attributes are
   * updated to reflect the removed control.
   *
   * @param op The original Unitary operation to clone.
   * @param operand The control qubit value to remove from `op`.
   * @param rewriter Pattern rewriter used to create and modify operations and
   *                 attributes.
   * @return UnitaryInterface A new Unitary operation equivalent to `op` but
   *         without the specified control operand.
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
   * @brief Transform a single basis-state control of a Unitary operation into
   * an scf.if driven by its measurement outcome.
   *
   * If the specified control operand originates from a MeasureOp, this rewrites
   * the Unitary by removing that control, creating a reduced Unitary operation
   * and an scf::IfOp that executes the reduced operation when the measurement
   * condition is satisfied and yields original inputs otherwise. The rewrite
   * replaces uses of the original results with the IfOp results and erases the
   * original Unitary op.
   *
   * @param op The Unitary operation to match and rewrite.
   * @param operand The control operand to match; must be produced by a
   * MeasureOp for the rewrite to succeed.
   * @param positive True if the control is positive (execute reduced op when
   * measurement bit is 1), false if negative.
   * @param rewriter The PatternRewriter used to construct and apply the
   * transformation.
   * @return UnitaryInterface The reduced Unitary operation with the specified
   * control removed if the rewrite was applied, `nullptr` otherwise.
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
   * @brief Attempt to process the first applicable control in a control set and
   * produce the resulting Unitary operation.
   *
   * Iterates the provided control operands and applies a single-control rewrite
   * when possible; stops after the first successful rewrite and returns the
   * updated operation.
   *
   * @param op The Unitary operation to process.
   * @param controls A collection of control operands to attempt processing.
   * @param isPositive True when the provided controls are positive controls,
   *                   false for negative controls.
   * @param rewriter PatternRewriter used to create and replace IR during the
   *                 transformation.
   * @return std::optional<UnitaryInterface> Containing the new UnitaryInterface
   *         with the processed control if a rewrite occurred, or `std::nullopt`
   *         if no control was processed.
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

  /**
   * @brief Iteratively removes basis-state controls from a Unitary operation by
   *        transforming matched controls into conditional `scf.if` blocks.
   *
   * The pattern repeatedly attempts to process positive controls, negative
   * controls, and—when the op is a diagonal gate with extra in-qubits—targets
   * treated as controls. When a control is processed the operation is updated
   * to the reduced form and the loop continues until no further controls can be
   * transformed.
   *
   * @param op The Unitary operation to match and rewrite; may be updated as
   *           controls are removed.
   * @param rewriter PatternRewriter used to perform rewrites and create new
   * ops.
   * @return mlir::LogicalResult `success(true)` if at least one control was
   *         transformed, `success(false)` if no changes were made.
   */
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
