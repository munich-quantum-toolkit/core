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
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace mqt::ir::opt {

/**
 * @brief This pattern is responsible for applying the "deferred measurement
 * principle", lifting measurements above controls.
 */
struct LiftMeasurementsAboveControlsPattern final
    : mlir::OpRewritePattern<MeasureOp> {

  explicit LiftMeasurementsAboveControlsPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(MeasureOp op,
                  mlir::PatternRewriter& rewriter) const override {
    const auto qubitVariable = op.getInQubit();
    auto* predecessor = qubitVariable.getDefiningOp();
    auto predecessorUnitary = mlir::dyn_cast<UnitaryInterface>(predecessor);

    if (!predecessorUnitary) {
      return mlir::failure();
    }

    if (std::find(predecessorUnitary.getOutQubits().begin(),
                  predecessorUnitary.getOutQubits().end(),
                  qubitVariable) != predecessorUnitary.getOutQubits().end()) {
      // The measured qubit is a target, not a control of the gate.
      return mlir::failure();
    }

    const auto correspondingInput =
        predecessorUnitary.getCorrespondingInput(qubitVariable);

    rewriter.replaceUsesWithIf(qubitVariable, correspondingInput,
                               [&](mlir::OpOperand& operand) {
                                 // We only replace the single use by the
                                 // measure op
                                 return operand.getOwner() == op;
                               });
    rewriter.replaceUsesWithIf(
        correspondingInput, op.getOutQubit(), [&](mlir::OpOperand& operand) {
          // We only replace the single use by the predecessor
          return operand.getOwner() == predecessorUnitary;
        });
    rewriter.replaceUsesWithIf(
        op.getOutQubit(), qubitVariable, [&](mlir::OpOperand& operand) {
          // All further uses of the measurement output now use the gate output
          return operand.getOwner() != predecessorUnitary;
        });
    rewriter.moveOpBefore(op, predecessor);

    return mlir::success();
  }
};

/**
 * @brief Populates the given pattern set with the
 * `LiftMeasurementsAboveControlsPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateLiftMeasurementsAboveControlsPatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<LiftMeasurementsAboveControlsPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
