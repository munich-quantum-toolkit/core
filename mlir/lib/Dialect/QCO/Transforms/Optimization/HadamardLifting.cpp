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

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_HADAMARDLIFTING
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

/**
 * @brief This method checks if two ranges contain of exactly the same
 * elements.
 *
 * This method checks if two ranges contain of exactly the same elements.
 *
 * @param range1 The first range.
 * @param range2 The second range.
 */
bool containRangesOfSameElements(const std::vector<Value>& range1,
                                 const std::vector<Value>& range2) {
  bool result = true;
  result &= range1.size() == range2.size();
  for (auto element : range1) {
    result &= std::ranges::find(range2, element) != range2.end();
  }
  return result;
}

/**
 * @brief This pattern changes the target of a controlled Pauli Z gate if a
 * controlled hadamard gate is it successor.
 * If all out qubits of Pauli Z are equal to all in qubits of Hadamard, we can
 * commute the gates and change Pauli Z to X. This is only possible if Hadamard
 * and Pauli act on the same qubit as target. If the target of the Pauli gate is
 * a ctrl at the hadamard and vice versa, we can change the target of Pauli Z to
 * the Hadamard's. This is done in this pattern.
 */
struct AdaptCtrldPauliZToLiftingPattern final : OpRewritePattern<CtrlOp> {

  explicit AdaptCtrldPauliZToLiftingPattern(MLIRContext* context)
      : OpRewritePattern(context) {}

  /**
   * @brief Checks if the target qubit of gate 1 is part of the ctrl qubits of
   * gate 2 and vice versa.
   *
   * This method checks if the output target qubit of gate 1 is used as control
   * qubit of gate 2. Additionally, it checks if the input target of gate 2 is
   * an output control of gate 2. Returns true if that is the case.
   * Must only be used on gates that have a single target qubit.
   *
   * @param gate1 First gate, predecessor of gate2.
   * @param gate2 Second gate, successor of gate1.
   * @return True if target qubit of gate1 is ctrl in gate2 and vice versa.
   * False otherwise.
   */
  static bool areTargetsControlsAtTheOtherGates(CtrlOp gate1, CtrlOp gate2) {
    const Value targetQubitGate2 = gate2.getInputTarget(0);
    const Value targetQubitGate1 = gate1.getOutputTarget(0);
    const auto inCtrlGate2 = gate2.getControlsIn();
    const auto outCtrlGate1 = gate1.getControlsOut();

    return std::find(inCtrlGate2.begin(), inCtrlGate2.end(),
                     targetQubitGate1) != inCtrlGate2.end() &&
           std::find(outCtrlGate1.begin(), outCtrlGate1.end(),
                     targetQubitGate2) != outCtrlGate1.end();
  }

  /**
   * @brief This method exchanges the position of two qubits acting on the same
   * gate.
   *
   * This method exchanges two qubits acting on the same gate. E.g. if qubit 1
   * is a target qubit and qubit 2 a control qubit, that is exchanged.
   *
   *
   * @param gate The gate both qubit1 and qubit2 belong to.
   * @param qubit1 First qubit, exchanged with second.
   * @param qubit2 Second qubit, exchanged with first.
   * @param rewriter The rewriter.
   */
  static void exchangeTwoQubitsAtGate(UnitaryOpInterface gate,
                                      const Value qubit1, const Value qubit2,
                                      PatternRewriter& rewriter) {
    auto temporary =
        rewriter.create<IdOp>(gate.getLoc(), gate.getInputTarget(0))
            .getResult();
    rewriter.replaceUsesWithIf(
        qubit1, temporary,
        [&](const OpOperand& operand) { return operand.getOwner() == gate; });
    rewriter.replaceUsesWithIf(qubit2, qubit1, [&](const OpOperand& operand) {
      return operand.getOwner() == gate;
    });
    rewriter.replaceUsesWithIf(
        temporary, qubit2,
        [&](const OpOperand& operand) { return operand.getOwner() == gate; });
  }

  /**
   * @brief Changes the target of a controlled Pauli Z gate if a
   * controlled hadamard gate is it successor.
   *
   * @param op The operation to match (only Pauli gates trigger the rewrite)
   * @param rewriter Pattern rewriter for applying transformations
   * @return success() if circuit was changed, failure() otherwise
   */
  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    // op needs to be a Pauli Z gate and controlled
    std::string opName = op.getBodyUnitary()->getName().stripDialect().str();
    if (op.getNumTargets() != 1 || opName != "z") {
      return failure();
    }

    // op needs to be in front of a controlled hadamard gate
    const auto& users = op->getUsers();
    if (users.empty()) {
      return failure();
    }
    auto user = *users.begin();
    if (user->getName().stripDialect().str() != "ctrl") {
      return failure();
    }
    auto hadamardGate = mlir::dyn_cast<CtrlOp>(user);
    if (hadamardGate.getNumTargets() != 1 ||
        hadamardGate.getBodyUnitary()->getName().stripDialect().str() != "h") {
      return failure();
    }

    const std::vector<Value> outputsOp(op.getOutputQubits().begin(),
                                       op.getOutputQubits().end());
    const std::vector<Value> inputsH(hadamardGate.getInputQubits().begin(),
                                     hadamardGate.getInputQubits().end());
    if (!containRangesOfSameElements(outputsOp, inputsH)) {
      return failure();
    }

    // If the target qubit of H is a ctrl in Z and vice versa, we can move Z's
    // target to H's target
    if (!areTargetsControlsAtTheOtherGates(op, hadamardGate)) {
      return failure();
    }

    // Put the Z target to the same qubit as the hadamard target is
    const Value originalInputTargetQubitZ = op.getInputTarget(0);
    const Value targetInputQubitHadamard = hadamardGate.getInputTarget(0);
    const Value newTargetInputQubitZ =
        op.getInputForOutput(targetInputQubitHadamard);

    exchangeTwoQubitsAtGate(op, originalInputTargetQubitZ, newTargetInputQubitZ,
                            rewriter);

    const Value newTargetInputQubitH =
        op.getOutputForInput(newTargetInputQubitZ);

    exchangeTwoQubitsAtGate(hadamardGate, targetInputQubitHadamard,
                            newTargetInputQubitH, rewriter);

    return success();
  }
};

/**
 * @brief This pattern is responsible for lifting Hadamard gates above Pauli
 * gates.
 *
 * This pattern swaps a Pauli gate with a Hadamard gate. This is done using the
 * commutation rules of Pauli and Hadamard gates, which are:
 * - X - H - = - H - Z -
 * - Y - H - = - H - Y -
 * - Z - H - = - H - X -
 * This is applied to uncontrolled gates and controlled ones, if the controls
 * are applied to the same qubits for both gates.
 */
struct LiftHadamardsAbovePauliGatesPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  explicit LiftHadamardsAbovePauliGatesPattern(MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  /**
   * This method checks whether the first and second controls are controlled by
   * the same qubits.
   *
   * @param firstCtrl The first (preceding) controlled gate.
   * @param secondCtrl The second (succeeding) controlled gate.
   * @return true if the controls are controlled by the same qubits.
   */
  static bool areControlsControlledBySameQubits(CtrlOp firstCtrl,
                                                CtrlOp secondCtrl) {
    const std::vector<Value> controlOutputsFirstGate(
        firstCtrl.getControlsOut().begin(), firstCtrl.getControlsOut().end());
    const std::vector<Value> controlInputsSecondGate(
        secondCtrl.getControlsIn().begin(), secondCtrl.getControlsIn().end());
    return containRangesOfSameElements(controlOutputsFirstGate,
                                       controlInputsSecondGate);
  }

  /**
   * @brief This method swaps a Pauli gate with a Hadamard gate.
   *
   * This method swaps a Pauli gate with a Hadamard gate. This is done using the
   * commutation rules of Pauli and Hadamard gates, which are:
   * - X - H - = - H - Z -
   * - Y - H - = - H - Y -
   * - Z - H - = - H - X -
   *
   * @param gate The Pauli gate.
   * @param hadamardGate The Hadamard gate.
   * @param rewriter The used rewriter.
   * @return success() if circuit was changed, failure() otherwise
   */
  static mlir::LogicalResult
  swapPauliWithHadamard(UnitaryOpInterface gate,
                        UnitaryOpInterface hadamardGate,
                        mlir::PatternRewriter& rewriter) {
    const auto gateName = gate->getName().stripDialect().str();
    const auto hadamardName = hadamardGate->getName().stripDialect().str();
    if (hadamardName != "h" ||
        (gateName != "x" && gateName != "y" && gateName != "z")) {
      return failure();
    }
    rewriter.setInsertionPoint(gate);
    rewriter.replaceOpWithNewOp<HOp>(gate, gate.getInputQubit(0));
    if (gateName == "x") {
      rewriter.setInsertionPoint(hadamardGate);
      rewriter.replaceOpWithNewOp<ZOp>(hadamardGate,
                                       hadamardGate.getInputQubit(0));
    } else if (gateName == "z") {
      rewriter.setInsertionPoint(hadamardGate);
      rewriter.replaceOpWithNewOp<XOp>(hadamardGate,
                                       hadamardGate.getInputQubit(0));
    } else {
      rewriter.setInsertionPoint(hadamardGate);
      rewriter.replaceOpWithNewOp<YOp>(hadamardGate,
                                       hadamardGate.getInputQubit(0));
    }
    return success();
  }

  /**
   * @brief Swaps controlled Hadamard and Pauli gate if they follow after each
   * other and are operated on by the same qubits.
   *
   * @param firstGate First controlled gate, needs to be a Pauli gate.
   * @param secondGate Second controlled gate, needs to be a Hadamard gate.
   * @param rewriter Pattern rewriter for applying transformations
   * @return success() if circuit was changed, failure() otherwise
   */
  static LogicalResult handleTwoSucceedingControls(CtrlOp firstGate,
                                                   CtrlOp secondGate,
                                                   PatternRewriter& rewriter) {
    if (firstGate.getNumTargets() != 1 || secondGate.getNumTargets() != 1 ||
        firstGate.getOutputTarget(0) != secondGate.getInputTarget(0) ||
        !areControlsControlledBySameQubits(firstGate, secondGate)) {
      return failure();
    }
    return swapPauliWithHadamard(firstGate.getBodyUnitary(),
                                 secondGate.getBodyUnitary(), rewriter);
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
    // op needs to be a Pauli gate
    std::string opName = op->getName().stripDialect().str();
    if (opName != "x" && opName != "y" && opName != "z" && opName != "ctrl") {
      return failure();
    }

    // op needs to be in front of a hadamard gate
    const auto& users = op->getUsers();
    if (users.empty()) {
      return failure();
    }
    auto user = *users.begin();
    auto userName = user->getName().stripDialect().str();
    if (userName != "h") {
      if (opName == "ctrl" && userName == "ctrl") {
        return handleTwoSucceedingControls(mlir::dyn_cast<CtrlOp>(*op),
                                           mlir::dyn_cast<CtrlOp>(user),
                                           rewriter);
      }
      return failure();
    }

    auto hadamardGate = mlir::dyn_cast<UnitaryOpInterface>(user);

    if (op.getNumControls() > 0 || hadamardGate.getNumControls() > 0 ||
        op.getNumTargets() != 1 || hadamardGate.getNumTargets() != 1 ||
        op.getOutputTarget(0) != hadamardGate.getInputTarget(0)) {
      return failure();
    }

    return swapPauliWithHadamard(op, hadamardGate, rewriter);
  }
};

/**
 * @brief This pattern remove an H gate between a CNOT and a measurement.
 *
 * If there is a Hadamard gate between the target qubit of a CNOT and a
 * measurement, we flip the CNOT and apply a hadamard gate to the incoming and
 * outcoming qubits. As H * H = id, the measurement is then the direct successor
 * of a CNOT ctrl, which is beneficial for the qubit reuse routine.
 * The procedure also works if there are additional ctrls. Only the target
 * and ctrl involved in the transformation get hadamard gates assigned.
 * For now, the involved ctrl to be flipped with the target is chosen randomly.
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
    auto temporary =
        rewriter.create<IdOp>(gate.getLoc(), gate.getInputTarget(0))
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
  static HOp addHadamardGatesBeforeGate(UnitaryOpInterface gate,
                                        std::vector<Value> inputQubits,
                                        PatternRewriter& rewriter) {
    HOp newHOP;
    for (Value inputQubit : inputQubits) {

      std::vector<Value> inQubits{inputQubit};
      std::vector<Type> outQubits{inputQubit.getType()};

      newHOP = rewriter.create<HOp>(gate->getLoc(), inQubits);

      rewriter.moveOpBefore(newHOP, gate);

      rewriter.replaceUsesWithIf(
          inputQubit, newHOP.getOutputTarget(0),
          [&](const OpOperand& operand) { return operand.getOwner() == gate; });
    }
    return newHOP;
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
  static HOp addHadamardGatesAfterGate(UnitaryOpInterface gate,
                                       const std::vector<Value>& outputQubits,
                                       PatternRewriter& rewriter) {
    HOp newHOp;
    for (Value outputQubit : outputQubits) {

      std::vector<Value> inQubit{outputQubit};
      std::vector<Type> outQubit{outputQubit.getType()};

      newHOp = rewriter.create<HOp>(gate->getLoc(), inQubit);

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
    auto hadamardGate = mlir::dyn_cast<UnitaryOpInterface>(predecessor);
    if (!hadamardGate || hadamardGate.getNumTargets() != 1 ||
        hadamardGate->getName().stripDialect().str() != "h") {
      return failure();
    }

    // The Hadamard gate must be successor of the target of a CNOT
    auto inQubitHadamard = hadamardGate.getInputQubit(0);
    predecessor = inQubitHadamard.getDefiningOp();
    auto cnotGate = mlir::dyn_cast<CtrlOp>(predecessor);
    if (!cnotGate || cnotGate.getNumTargets() != 1 ||
        cnotGate.getBodyUnitary()->getName().stripDialect().str() != "x" ||
        cnotGate.getOutputTarget(0) != inQubitHadamard) {
      return failure();
    }

    // Remove the Hadamard gate
    for (auto outQubit : hadamardGate.getOutputQubits()) {
      rewriter.replaceAllUsesWith(outQubit,
                                  hadamardGate.getInputForOutput(outQubit));
    }
    rewriter.eraseOp(hadamardGate);

    // Add Hadamard gates to the other in and output gates of cnot
    const std::vector<Value> relevantInputQubitsForHadamard{
        cnotGate.getInputTarget(0), cnotGate.getInputControl(0)};
    addHadamardGatesBeforeGate(cnotGate, relevantInputQubitsForHadamard,
                               rewriter);

    const std::vector<Value> relevantOutputQubitsForHadamard{
        cnotGate.getOutputForInput(cnotGate.getInputControl(0))};
    HOp newHOPAfterCtrl = addHadamardGatesAfterGate(
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
    patterns.add<AdaptCtrldPauliZToLiftingPattern>(patterns.getContext());
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