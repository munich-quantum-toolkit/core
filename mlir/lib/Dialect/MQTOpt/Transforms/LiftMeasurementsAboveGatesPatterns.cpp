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

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/UseDefLists.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <string>
#include <unordered_set>

namespace mqt::ir::opt {

static const std::unordered_set<std::string> INVERTING_GATES = {"x", "y"};
static const std::unordered_set<std::string> DIAGONAL_GATES = {
    "i", "z", "s", "sdg", "t", "tdg", "p", "rz"};

void swapGateWithMeasurement(UnitaryInterface gate, MeasureOp measurement,
                             mlir::PatternRewriter& rewriter) {
  auto measurementInput = measurement.getInQubit();
  auto gateInput = gate.getCorrespondingInput(measurementInput);
  rewriter.replaceUsesWithIf(measurementInput, gateInput,
                             [&](mlir::OpOperand& operand) {
                               // We only replace the single use by the
                               // measure op
                               return operand.getOwner() == measurement;
                             });
  rewriter.replaceUsesWithIf(gateInput, measurement.getOutQubit(),
                             [&](mlir::OpOperand& operand) {
                               // We only replace the single use by the
                               // predecessor
                               return operand.getOwner() == gate;
                             });
  rewriter.replaceUsesWithIf(measurement.getOutQubit(), measurementInput,
                             [&](mlir::OpOperand& operand) {
                               // All further uses of the measurement output now
                               // use the gate output
                               return operand.getOwner() != gate;
                             });
  rewriter.moveOpBefore(measurement, gate);
}

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
    const auto qubitVariable = op.getInQubit();
    auto* predecessor = qubitVariable.getDefiningOp();
    const auto name = predecessor->getName().stripDialect().str();

    auto predecessorUnitary = mlir::dyn_cast<UnitaryInterface>(predecessor);

    if (!predecessorUnitary) {
      return mlir::failure();
    }

    if (DIAGONAL_GATES.count(name) == 1) {
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
      return mlir::isa<ResetOp>(user) || mlir::isa<DeallocQubitOp>(user);
    });
  }

  mlir::LogicalResult
  matchAndRewrite(MeasureOp op,
                  mlir::PatternRewriter& rewriter) const override {
    if (!outputQubitRemainsUnused(op.getOutQubit())) {
      return mlir::failure(); // if the qubit is still used after the
                              // measurement, we cannot lift it above the gate.
    }
    const auto qubitVariable = op.getInQubit();
    auto* predecessor = qubitVariable.getDefiningOp();
    const auto name = predecessor->getName().stripDialect().str();

    auto predecessorUnitary = mlir::dyn_cast<UnitaryInterface>(predecessor);

    if (!predecessorUnitary) {
      return mlir::failure();
    }

    if (INVERTING_GATES.count(name) == 1 &&
        predecessorUnitary.getAllInQubits().size() == 1) {
      swapGateWithMeasurement(predecessorUnitary, op, rewriter);
      rewriter.setInsertionPointAfter(op);
      const mlir::Value trueConstant = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getBoolAttr(true));
      auto inversion = rewriter.create<mlir::arith::XOrIOp>(
          op.getLoc(), op.getOutBit(), trueConstant);
      // We need `replaceUsesWithIf` so that we can replace all uses except for
      // the one use that defines the inverted bit.
      rewriter.replaceUsesWithIf(op.getOutBit(), inversion.getResult(),
                                 [&](mlir::OpOperand& operand) {
                                   return operand.getOwner() != inversion;
                                 });
      return mlir::success();
    }

    return mlir::failure();
  }
};

/**
 * @brief Populates the given pattern set with the
 * `LiftMeasurementsAbovePhaseGatesPattern` and
 * `LiftMeasurementsAboveInvertingGatesPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateLiftMeasurementsAboveGatesPatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<LiftMeasurementsAbovePhaseGatesPattern>(patterns.getContext());
  patterns.add<LiftMeasurementsAboveInvertingGatesPattern>(
      patterns.getContext());
}

} // namespace mqt::ir::opt
