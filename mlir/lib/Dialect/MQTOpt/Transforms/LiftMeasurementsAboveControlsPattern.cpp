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
#include "mlir/Dialect/MQTOpt/Transforms/LiftMeasurementsPasses.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <llvm/ADT/STLExtras.h>
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

  /**
       * @brief Constructs a rewrite pattern that lifts measurement operations above their control gates.
       *
       * @param context MLIR context used to initialize the base OpRewritePattern and access MLIR types and operations.
       */
      explicit LiftMeasurementsAboveControlsPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  /**
   * @brief Attempts to lift a measurement above its controlling unitary gate.
   *
   * Applies when the measured qubit is produced by a unitary operation and that
   * qubit is used as a control (not an output target) of the unitary; in that
   * case the pattern swaps the gate with the measurement to defer the
   * measurement.
   *
   * @return mlir::LogicalResult `mlir::success()` if the measurement was lifted
   * and the rewrite applied, `mlir::failure()` otherwise.
   */
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
 * @brief Add a rewrite pattern that lifts measurement operations above their control gates.
 *
 * Inserts an instance of LiftMeasurementsAboveControlsPattern into the provided pattern set so it will be considered during pattern-driven rewrites.
 *
 * @param patterns The rewrite pattern set to populate.
 */
void populateLiftMeasurementsAboveControlsPatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<LiftMeasurementsAboveControlsPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt