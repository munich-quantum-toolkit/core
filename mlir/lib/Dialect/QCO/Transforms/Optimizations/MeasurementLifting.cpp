/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

//
// Created by damian on 5/13/26.
//

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_MEASUREMENTLIFTING
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

/**
 * @brief Checks if the given operation is an inverting gate.
 * @param op The operation to check.
 * @return True if the operation is an inverting gate, false otherwise.
 */
static bool isInverting(Operation* op) { return isa<XOp, YOp>(op); }

/**
 * @brief Checks if the given operation is a diagonal gate.
 * @param op The operation to check.
 * @return True if the operation is a diagonal gate, false otherwise.
 */
static bool isDiagonal(Operation* op) {
  return isa<ZOp, SOp, TOp, POp, RZOp, SdgOp, TdgOp, IdOp>(op);
}

/**
 * @brief This method swaps a gate with a measurement.
 * @param gate The gate to swap.
 * @param measurement The measurement to swap.
 * @param rewriter The used rewriter.
 */
static void swapGateWithMeasurement(UnitaryOpInterface gate,
                                    MeasureOp measurement,
                                    mlir::PatternRewriter& rewriter) {
  auto measurementInput = measurement.getQubitIn();
  auto gateInput = gate.getInputForOutput(measurementInput);
  rewriter.replaceUsesWithIf(measurementInput, gateInput,
                             [&](mlir::OpOperand& operand) {
                               // We only replace the single use by the
                               // measure op
                               return operand.getOwner() == measurement;
                             });
  rewriter.replaceUsesWithIf(gateInput, measurement.getQubitOut(),
                             [&](mlir::OpOperand& operand) {
                               // We only replace the single use by the
                               // predecessor
                               return operand.getOwner() == gate;
                             });
  rewriter.replaceUsesWithIf(measurement.getQubitOut(), measurementInput,
                             [&](mlir::OpOperand& operand) {
                               // All further uses of the measurement output now
                               // use the gate output
                               return operand.getOwner() != gate;
                             });
  rewriter.moveOpBefore(measurement, gate);
}

namespace {
/**
 * @brief This pattern is responsible for lifting measurements above any phase
 * gates.
 */
struct LiftMeasurementsAbovePhaseGatesPattern final
    : mlir::OpRewritePattern<MeasureOp> {

  explicit LiftMeasurementsAbovePhaseGatesPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(MeasureOp op,
                  mlir::PatternRewriter& rewriter) const override {
    const auto qubitVariable = op.getQubitIn();
    auto* predecessor = qubitVariable.getDefiningOp();

    auto predecessorUnitary = mlir::dyn_cast<UnitaryOpInterface>(predecessor);

    if (!predecessorUnitary) {
      return mlir::failure();
    }

    if (isDiagonal(predecessor)) {
      swapGateWithMeasurement(predecessorUnitary, op, rewriter);
      return mlir::success();
    }

    return mlir::failure();
  }
};

/**
 * @brief This pattern is responsible for lifting measurements above any
 * non-phase gates.
 */
struct LiftMeasurementsAboveInvertingGatesPattern final
    : mlir::OpRewritePattern<MeasureOp> {

  explicit LiftMeasurementsAboveInvertingGatesPattern(
      mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  /**
   * @brief Checks if the given qubit is not used anymore.
   * @param outQubit The output qubit to check.
   * @return True if all users are resets/deallocs, false otherwise.
   */
  static bool outputQubitRemainsUnused(mlir::Value outQubit) {
    return llvm::all_of(outQubit.getUsers(), [](mlir::Operation* user) {
      return mlir::isa<ResetOp>(user) || mlir::isa<SinkOp>(user);
    });
  }

  mlir::LogicalResult
  matchAndRewrite(MeasureOp op,
                  mlir::PatternRewriter& rewriter) const override {
    if (!outputQubitRemainsUnused(op.getQubitOut())) {
      return mlir::failure(); // if the qubit is still used after the
                              // measurement, we cannot lift it above the gate.
    }
    const auto qubitVariable = op.getQubitIn();
    auto* predecessor = qubitVariable.getDefiningOp();

    auto predecessorUnitary = mlir::dyn_cast<UnitaryOpInterface>(predecessor);

    if (!predecessorUnitary) {
      return mlir::failure();
    }

    if (isInverting(predecessor) &&
        predecessorUnitary.getInputQubits().size() == 1) {
      swapGateWithMeasurement(predecessorUnitary, op, rewriter);
      rewriter.setInsertionPointAfter(op);
      const mlir::Value trueConstant = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getBoolAttr(true));
      auto inversion = rewriter.create<mlir::arith::XOrIOp>(
          op.getLoc(), op.getResult(), trueConstant);
      // We need `replaceUsesWithIf` so that we can replace all uses except for
      // the one use that defines the inverted bit.
      rewriter.replaceUsesWithIf(op.getResult(), inversion.getResult(),
                                 [&](mlir::OpOperand& operand) {
                                   return operand.getOwner() != inversion;
                                 });
      return mlir::success();
    }

    return mlir::failure();
  }
};

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
    const auto qubitVariable = op.getQubitIn();
    auto* predecessor = qubitVariable.getDefiningOp();
    auto predecessorUnitary = mlir::dyn_cast<UnitaryOpInterface>(predecessor);

    if (!predecessorUnitary) {
      return mlir::failure();
    }

    if (llvm::find(predecessorUnitary.getOutputQubits(), qubitVariable) !=
        predecessorUnitary.getOutputQubits().end()) {
      // The measured qubit is a target, not a control of the gate.
      return mlir::failure();
    }

    swapGateWithMeasurement(predecessorUnitary, op, rewriter);

    return mlir::success();
  }
};

/**
 * @brief Pass raises Measurements above controlled and uncontrolled gates
 * gates.
 */
struct MeasurementLifting final
    : impl::MeasurementLiftingBase<MeasurementLifting> {
  using MeasurementLiftingBase::MeasurementLiftingBase;

protected:
  void runOnOperation() override {
    const auto op = getOperation();
    auto* ctx = &getContext();

    // Define the set of patterns to use.
    RewritePatternSet patterns(ctx);
    patterns.add<LiftMeasurementsAboveControlsPattern>(patterns.getContext());
    patterns.add<LiftMeasurementsAboveInvertingGatesPattern>(
        patterns.getContext());
    patterns.add<LiftMeasurementsAbovePhaseGatesPattern>(patterns.getContext());

    // Apply patterns in an iterative and greedy manner.
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco
