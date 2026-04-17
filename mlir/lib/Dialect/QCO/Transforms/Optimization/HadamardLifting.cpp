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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cmath>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_HADAMARDLIFTING
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

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
    auto* op = gate.getOperation();
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

    // op needs to be in front of a Hadamard gate
    auto hadamardGate = llvm::dyn_cast<HOp>(*op->getUsers().begin());

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
   * @brief This method swaps two operand usages in an operation.
   *
   * @param op The operation on with the operand usage should be swapped.
   * @param a The operand value to be swapped with b.
   * @param b The operand value to be swapped with a.
   */
  static void swapOperandsInOp(Operation* op, const Value a, const Value b) {
    for (OpOperand& operand :
         llvm::make_filter_range(op->getOpOperands(), [&](const OpOperand& o) {
           const Value valueOfOperand = o.get();
           return valueOfOperand == a || valueOfOperand == b;
         })) {
      const bool operandValueIsA = operand.get() == a;
      operand.set(operandValueIsA ? b : a);
    }
  }

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
  static void swapQubits(CtrlOp gate, const Value inputQubit1,
                         const Value inputQubit2, Operation* succeedingOp1,
                         Operation* succeedingOp2, PatternRewriter& rewriter) {
    const Value outputQubit1 = gate.getOutputForInput(inputQubit1);
    const Value outputQubit2 = gate.getOutputForInput(inputQubit2);

    rewriter.modifyOpInPlace(
        gate, [&] { swapOperandsInOp(gate, inputQubit1, inputQubit2); });

    rewriter.modifyOpInPlace(succeedingOp1, [&] {
      swapOperandsInOp(succeedingOp1, outputQubit1, outputQubit2);
    });

    rewriter.modifyOpInPlace(succeedingOp2, [&] {
      swapOperandsInOp(succeedingOp2, outputQubit1, outputQubit2);
    });
  }

  /**
   * @brief This method adds Hadamard gates before a given gate.
   *
   * @param gate The gate before which Hadamard gates should be applied.
   * @param inputQubits The input qubits of gate before which Hadamard gates
   * should be applied.
   * @param rewriter The used rewriter.
   * @returns One of the created Hadamard gates.
   */
  static HOp addHadamardGatesBeforeGate(const CtrlOp gate,
                                        const ValueRange inputQubits,
                                        PatternRewriter& rewriter) {
    HOp newHOp;
    for (const Value inputQubit : inputQubits) {

      const ValueRange inQubits(inputQubit);

      newHOp = HOp::create(rewriter, gate->getLoc(), inQubits);

      rewriter.moveOpBefore(newHOp, gate);

      rewriter.modifyOpInPlace(gate, [&] {
        swapOperandsInOp(gate, inputQubit, newHOp.getOutputTarget(0));
      });
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
   * @returns One of the created Hadamard gates.
   */
  static HOp addHadamardGatesAfterGate(const CtrlOp gate,
                                       const ValueRange outputQubits,
                                       PatternRewriter& rewriter) {
    HOp newHOp;
    for (Value outputQubit : outputQubits) {

      const ValueRange inQubit(outputQubit);

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
   * @brief This pattern removes an H gate between a CNOT and a measurement,
   * flips the CNOT and adds Hadamard gates before and after the new target and
   * before the new control.
   *
   * @param op The operation to match (only measurements with an uncontrolled
   * Hadamard gate before that trigger the rewrite)
   * @param rewriter Pattern rewriter for applying transformations
   * @return success() if circuit was changed, failure() otherwise
   */
  LogicalResult matchAndRewrite(MeasureOp op,
                                PatternRewriter& rewriter) const override {
    // A Hadamard gate needs to be in front of the measurement
    const auto qubitInMeasurement = op.getQubitIn();
    auto* predecessor = qubitInMeasurement.getDefiningOp();
    auto hadamardGate = llvm::dyn_cast<HOp>(predecessor);
    if (!hadamardGate) {
      return failure();
    }

    // The Hadamard gate must be successor of the target of a CNOT
    const auto inQubitHadamard = hadamardGate.getInputQubit(0);
    predecessor = inQubitHadamard.getDefiningOp();
    auto cnotGate = llvm::dyn_cast<CtrlOp>(predecessor);
    if (!cnotGate || cnotGate.getNumTargets() != 1 ||
        cnotGate.getOutputTarget(0) != inQubitHadamard ||
        llvm::dyn_cast<XOp>(cnotGate.getBodyUnitary()) == nullptr) {
      return failure();
    }
    // Determine the index of the control that will become the new target. The
    // control must not be succeeded by a measurement.
    unsigned int controlIndex = 0;
    for (unsigned int i = 0; i < cnotGate.getNumControls(); i++) {
      if (llvm::dyn_cast<MeasureOp>(
              *cnotGate.getOutputControl(i).getUsers().begin())) {
        if (i == cnotGate.getNumControls() - 1) {
          return failure();
        }
      } else {
        controlIndex = i;
        break;
      }
    }

    // Remove the Hadamard gate
    for (const auto outQubit : hadamardGate.getOutputQubits()) {
      rewriter.replaceAllUsesWith(outQubit,
                                  hadamardGate.getInputForOutput(outQubit));
    }
    rewriter.eraseOp(hadamardGate);

    // Add Hadamard gates to the other in- and output gates of CNOT
    const ValueRange relevantInputQubitsForHadamard(
        {cnotGate.getInputTarget(0), cnotGate.getInputControl(controlIndex)});
    addHadamardGatesBeforeGate(cnotGate, relevantInputQubitsForHadamard,
                               rewriter);

    const HOp newHOPAfterCtrl = addHadamardGatesAfterGate(
        cnotGate,
        cnotGate.getOutputForInput(cnotGate.getInputControl(controlIndex)),
        rewriter);

    // Flip CNOT targets and ctrl
    swapQubits(cnotGate, cnotGate.getInputControl(controlIndex),
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
