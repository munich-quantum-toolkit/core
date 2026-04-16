/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <math.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace mlir::qco {

#define GEN_PASS_DEF_HADAMARDLIFTING
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

/**
 * @brief This method checks if two ranges contain of exactly the same
 * elements.
 *
 * This method checks if two ranges contain of exactly the same elements.
 *
 * @param range1 The first range.
 * @param range2 The second range.
 */
static bool containRangesOfSameElements(const std::vector<Value>& range1,
                                        const std::vector<Value>& range2) {
  return range1.size() == range2.size() &&
         std::ranges::is_permutation(range1, range2);
}

namespace {

/**
 * @brief This pattern is responsible for lifting Hadamard gates above Pauli
 * gates.
 *
 * This pattern swaps a Pauli gate with a Hadamard gate. This is done using the
 * commutation rules of Pauli and Hadamard gates, which are:
 * - X - H - = - H - Z -
 * - Y - H - = - H - Y - G(pi) -
 * - Z - H - = - H - X -
 * This is applied to uncontrolled gates.
 * In case of Pauli-Y, a global phase is applied, as HY = -YH.
 */
struct LiftHadamardsAbovePauliGatesPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  explicit LiftHadamardsAbovePauliGatesPattern(MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  /**
   * @brief This method swaps a Pauli gate with a Hadamard gate.
   *
   * This method swaps a Pauli gate with a Hadamard gate. This is done using the
   * commutation rules of Pauli and Hadamard gates, which are:
   * - X - H - = - H - Z -
   * - Y - H - = - H - Y - gPhase(pi) -
   * - Z - H - = - H - X -
   *
   * @param gate The Pauli gate.
   * @param hadamardGate The Hadamard gate.
   * @param rewriter The used rewriter.
   * @return success() if circuit was changed, failure() otherwise
   */
  static LogicalResult swapPauliWithHadamard(UnitaryOpInterface gate,
                                             HOp hadamardGate,
                                             PatternRewriter& rewriter) {
    auto op = gate.getOperation();
    return TypeSwitch<Operation*, LogicalResult>(op)
        .Case<XOp>([&](auto) {
          rewriter.replaceOpWithNewOp<HOp>(gate, gate.getInputQubit(0));
          rewriter.replaceOpWithNewOp<ZOp>(hadamardGate,
                                           hadamardGate.getInputQubit(0));
          return success();
        })
        .Case<ZOp>([&](auto) {
          rewriter.replaceOpWithNewOp<HOp>(gate, gate.getInputQubit(0));
          rewriter.replaceOpWithNewOp<XOp>(hadamardGate,
                                           hadamardGate.getInputQubit(0));
          return success();
        })
        .Case<YOp>([&](auto) {
          rewriter.replaceOpWithNewOp<HOp>(gate, gate.getInputQubit(0));
          rewriter.replaceOpWithNewOp<YOp>(hadamardGate,
                                           hadamardGate.getInputQubit(0));
          GPhaseOp::create(rewriter, hadamardGate.getLoc(), M_PI);
          return success();
        })
        .Default([&](auto) { return failure(); });
  }

  /**
   * @brief Lifts Hadamard gates in front of Pauli gates.
   *
   * @param op The operation to match (only Pauli gates trigger the rewrite)
   * @param rewriter Pattern rewriter for applying transformations
   * @return success() if circuit was changed, failure() otherwise
   */
  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    // op needs to be an uncontrolled Pauli gate
    if (!llvm::isa<XOp>(op) && !llvm::isa<YOp>(op) && !llvm::isa<ZOp>(op)) {
      return failure();
    }

    // op needs to be in front of a hadamard gate
    const auto& users = op->getUsers();
    if (users.empty()) {
      return failure();
    }
    auto* const user = *users.begin();
    auto hadamardGate = llvm::dyn_cast<HOp>(user);

    if (!hadamardGate ||
        op.getOutputTarget(0) != hadamardGate.getInputTarget(0)) {
      return failure();
    }

    return swapPauliWithHadamard(op, hadamardGate, rewriter);
  }
};

/**
 * @brief This pattern removes an H gate between a CNOT and a measurement, flips
 * the CNOT and adds Hadamard gates before and after the new target and before
 * the new control.
 *
 * If there is a Hadamard gate between the target qubit of a CNOT and a
 * measurement, we flip the CNOT and apply a hadamard gate to the incoming and
 * outcoming qubits. As H * H = id, the measurement is then the direct successor
 * of a CNOT control, which is beneficial for the qubit reuse routine. After the
 * application of LiftHadamardAboveCNOTPattern, a measurement will follow
 * directly after a control. In that case, measurement lifting (a routine of
 * qubit reuse) can remove the multi-qubit gate by lifting the measurement in
 * front of the control and changing the qubit-controlled Pauli-X to a
 * classically controlled Pauli-X.
 *
 * The procedure also works if there are additional ctrls. Only the target
 * and ctrl involved in the transformation get hadamard gates assigned.
 * The involved ctrl to be flipped with the target is chosen randomly.
 */
struct LiftHadamardAboveCNOTPattern final : OpRewritePattern<MeasureOp> {

  explicit LiftHadamardAboveCNOTPattern(MLIRContext* context)
      : OpRewritePattern(context) {}

  /**
   * @brief This method swaps two qubits on a gate.
   *
   * This method swaps two qubits on a gate. Input and output are exchanged.
   *
   * @param gate The gate that the qubits belong to.
   * @param inputQubit1 The input qubit of the qubit to be exchanged with 2.
   * @param inputQubit2 The input qubit of the qubit to be exchanged with 1.
   * @param succeedingOp1 The operation succeeding gate on the corresponding
   * output of inputQubit1.
   * @param succeedingOp2 The operation succeeding gate on the corresponding
   * output of inputQubit2.
   * @param rewriter The used rewriter.
   */
  static void swapQubits(UnitaryOpInterface gate, const Value inputQubit1,
                         const Value inputQubit2,
                         const Operation* succeedingOp1,
                         const Operation* succeedingOp2,
                         PatternRewriter& rewriter) {
    const Value outputQubit1 = gate.getOutputForInput(inputQubit1);
    const Value outputQubit2 = gate.getOutputForInput(inputQubit2);
    const auto temporary =
        IdOp::create(rewriter, gate.getLoc(), gate.getInputTarget(0))
            .getResult();

    rewriter.replaceUsesWithIf(outputQubit1, temporary,
                               [&](const OpOperand& operand) {
                                 return operand.getOwner() == gate ||
                                        operand.getOwner() == succeedingOp1 ||
                                        operand.getOwner() == succeedingOp2;
                               });
    rewriter.replaceUsesWithIf(outputQubit2, outputQubit1,
                               [&](const OpOperand& operand) {
                                 return operand.getOwner() == gate ||
                                        operand.getOwner() == succeedingOp1 ||
                                        operand.getOwner() == succeedingOp2;
                               });
    rewriter.replaceUsesWithIf(temporary, outputQubit2,
                               [&](const OpOperand& operand) {
                                 return operand.getOwner() == gate ||
                                        operand.getOwner() == succeedingOp1 ||
                                        operand.getOwner() == succeedingOp2;
                               });

    rewriter.replaceUsesWithIf(
        inputQubit1, temporary,
        [&](const OpOperand& operand) { return operand.getOwner() == gate; });
    rewriter.replaceUsesWithIf(
        inputQubit2, inputQubit1,
        [&](const OpOperand& operand) { return operand.getOwner() == gate; });
    rewriter.replaceUsesWithIf(
        temporary, inputQubit2,
        [&](const OpOperand& operand) { return operand.getOwner() == gate; });
  }

  /**
   * @brief This method adds hadamrad gates before a given gate.
   *
   * @param gate The gate before which hadamard gates should be applied.
   * @param inputQubits The input qubits of gate before which hadamard gates
   * should be applied.
   * @param rewriter The used rewriter.
   * @returns One of the created hadamard gates.
   */
  static HOp addHadamardGatesBeforeGate(const UnitaryOpInterface gate,
                                        const std::vector<Value>& inputQubits,
                                        PatternRewriter& rewriter) {
    HOp newHOp;
    for (const Value inputQubit : inputQubits) {

      std::vector inQubits{inputQubit};

      newHOp = HOp::create(rewriter, gate->getLoc(), inQubits);

      rewriter.moveOpBefore(newHOp, gate);

      rewriter.replaceUsesWithIf(
          inputQubit, newHOp.getOutputTarget(0),
          [&](const OpOperand& operand) { return operand.getOwner() == gate; });
    }
    return newHOp;
  }

  /**
   * @brief This method adds Hadamard gates after a given gate.
   *
   * @param gate The gate after which Hadamard gates should be applied.
   * @param outputQubits The output qubits of gate after which Hadamard gates
   * should be applied.
   * @param rewriter The used rewriter.
   * @returns One of the created hadamard gates.
   */
  static HOp addHadamardGatesAfterGate(const UnitaryOpInterface gate,
                                       const std::vector<Value>& outputQubits,
                                       PatternRewriter& rewriter) {
    HOp newHOp;
    for (Value outputQubit : outputQubits) {

      std::vector inQubit{outputQubit};
      std::vector outQubit{outputQubit.getType()};

      newHOp = HOp::create(rewriter, gate->getLoc(), inQubit);

      rewriter.moveOpAfter(newHOp, gate);

      rewriter.replaceUsesWithIf(
          newHOp.getInputTarget(0), newHOp.getOutputTarget(0),
          [&](const OpOperand& operand) {
            return operand.getOwner() != gate && operand.getOwner() != newHOp;
          });
    }
    return newHOp;
  }

  /**
   * @brief This pattern remove an H gate between a CNOT and a measurement.
   *
   * @param op The operation to match (only uncontrolled Hadamard gates trigger
   * the rewrite)
   * @param rewriter Pattern rewriter for applying transformations
   * @return success() if circuit was changed, failure() otherwise
   */
  LogicalResult matchAndRewrite(MeasureOp op,
                                PatternRewriter& rewriter) const override {
    // A Hadamard gate needs to be in front of the measurement
    const auto qubitInMeasurement = op.getQubitIn();
    auto* predecessor = qubitInMeasurement.getDefiningOp();
    auto hadamardGate = llvm::dyn_cast<UnitaryOpInterface>(predecessor);
    if (!hadamardGate || hadamardGate.getNumTargets() != 1 ||
        hadamardGate->getName().stripDialect().str() != "h") {
      return failure();
    }

    // The Hadamard gate must be successor of the target of a CNOT
    const auto inQubitHadamard = hadamardGate.getInputQubit(0);
    predecessor = inQubitHadamard.getDefiningOp();
    auto cnotGate = llvm::dyn_cast<CtrlOp>(predecessor);
    if (!cnotGate || cnotGate.getNumTargets() != 1 ||
        cnotGate.getBodyUnitary()->getName().stripDialect().str() != "x" ||
        cnotGate.getOutputTarget(0) != inQubitHadamard) {
      return failure();
    }

    // Remove the Hadamard gate
    for (const auto outQubit : hadamardGate.getOutputQubits()) {
      rewriter.replaceAllUsesWith(outQubit,
                                  hadamardGate.getInputForOutput(outQubit));
    }
    rewriter.eraseOp(hadamardGate);

    // Add Hadamard gates to the other in and output gates of cnot
    const std::vector relevantInputQubitsForHadamard{
        cnotGate.getInputTarget(0), cnotGate.getInputControl(0)};
    addHadamardGatesBeforeGate(cnotGate, relevantInputQubitsForHadamard,
                               rewriter);

    const std::vector relevantOutputQubitsForHadamard{
        cnotGate.getOutputForInput(cnotGate.getInputControl(0))};
    const HOp newHOPAfterCtrl = addHadamardGatesAfterGate(
        cnotGate, relevantOutputQubitsForHadamard, rewriter);

    // Flip CNOT targets and ctrl
    swapQubits(cnotGate, cnotGate.getInputControl(0),
               cnotGate.getInputTarget(0), op, newHOPAfterCtrl, rewriter);

    return success();
  }
};

/**
 * @brief Pass raises Hadamard gates above controlled and uncontrolled Pauli
 * gates.
 */
struct HadamardLifting final : impl::HadamardLiftingBase<HadamardLifting> {
  using HadamardLiftingBase::HadamardLiftingBase;

protected:
  void runOnOperation() override {
    const auto op = getOperation();
    auto* ctx = &getContext();

    // Define the set of patterns to use.
    RewritePatternSet patterns(ctx);
    patterns.add<LiftHadamardsAbovePauliGatesPattern>(patterns.getContext());
    patterns.add<LiftHadamardAboveCNOTPattern>(patterns.getContext());

    // Apply patterns in an iterative and greedy manner.
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco
