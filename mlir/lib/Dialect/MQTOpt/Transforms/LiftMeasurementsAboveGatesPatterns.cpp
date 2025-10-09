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
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/UseDefLists.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <string>
#include <unordered_set>

namespace mqt::ir::opt {

static const std::unordered_set<std::string> INVERTING_GATES = {"x", "y"};
static const std::unordered_set<std::string> DIAGONAL_GATES = {
    "i", "z", "s", "sdg", "t", "tdg", "p", "rz"};

/**
 * @brief Swap a unitary gate with a subsequent measurement, updating uses and
 * moving the measurement before the gate.
 *
 * Rewires the qubit/value connections so that the measurement takes the gate's
 * input and the gate consumes the measurement's output, then moves the
 * measurement operation to precede the gate.
 *
 * @param gate The unitary operation whose position is exchanged with the
 * measurement.
 * @param measurement The measurement operation to be lifted above the gate.
 * @param rewriter PatternRewriter used to perform use-replacements and to move
 * the measurement op.
 */
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

  /**
   * @brief Construct a pattern that lifts measurements above phase (diagonal)
   * gates.
   *
   * @param context MLIR context used to initialize the underlying rewrite
   * pattern.
   */
  explicit LiftMeasurementsAbovePhaseGatesPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  /**
   * @brief Attempts to lift a measurement operation above a preceding diagonal
   * phase gate.
   *
   * If the measurement's input is defined by a gate whose name is listed in
   * DIAGONAL_GATES, rewrites the IR so the measurement is moved before that
   * gate.
   *
   * @param op The MeasureOp to match and possibly rewrite.
   * @param rewriter The PatternRewriter used to apply the transformation.
   * @return mlir::LogicalResult `success` if the measurement was lifted,
   * `failure` otherwise.
   */
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

  /**
   * @brief Constructs a LiftMeasurementsAboveInvertingGatesPattern.
   *
   * Initializes the underlying OpRewritePattern using the provided MLIR
   * context.
   *
   * @param context MLIRContext used to register and apply the rewrite pattern.
   */
  explicit LiftMeasurementsAboveInvertingGatesPattern(
      mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  /**
   * @brief Determines whether all users of the given qubit are resets or
   * deallocations.
   *
   * @param outQubit The output qubit value to inspect.
   * @return `true` if every user is a `ResetOp` or `DeallocQubitOp`, `false`
   * otherwise.
   */
  static bool outputQubitRemainsUnused(mlir::Value outQubit) {
    return llvm::all_of(outQubit.getUsers(), [](mlir::Operation* user) {
      return mlir::isa<ResetOp>(user) || mlir::isa<DeallocQubitOp>(user);
    });
  }

  /**
   * Attempts to lift a measurement above a single-qubit inverting gate by
   * swapping the measurement with the gate and replacing the measured bit with
   * its inversion.
   *
   * This match checks that the measurement's output qubit has no remaining
   * users other than resets/deallocations, that the preceding operation is a
   * single-qubit unitary whose name is listed in INVERTING_GATES, and then
   * performs the swap. After swapping, a boolean `true` constant is XORed with
   * the measurement result to produce the inverted bit, and uses of the
   * original measured bit are replaced (excluding the internal use that defines
   * the inversion).
   *
   * @param op The MeasureOp to match and potentially rewrite.
   * @param rewriter PatternRewriter used to perform replacements and
   * insertions.
   * @return `mlir::success()` if the measurement was lifted and the bit
   * replaced with its inverted value, `mlir::failure()` otherwise.
   */
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
