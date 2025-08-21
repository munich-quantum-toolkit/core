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

    if (llvm::find(predecessorUnitary.getOutQubits(), qubitVariable) !=
        predecessorUnitary.getOutQubits().end()) {
      // The measured qubit is a target, not a control of the gate.
      return mlir::failure();
    }

    swapGateWithMeasurement(predecessorUnitary, op, rewriter);

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
