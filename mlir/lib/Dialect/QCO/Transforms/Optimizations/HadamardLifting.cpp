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
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <numbers>
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
 * - Y - H - = - H - Y -, while adding a gPhase(pi)
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
   * - Y - H - = - H - Y -, while adding a gPhase(pi)
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
          auto yGate = rewriter.replaceOpWithNewOp<YOp>(
              hadamardGate, hadamardGate.getInputQubit(0));
          GPhaseOp::create(rewriter, yGate.getLoc(), std::numbers::pi);
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
    if (!isa<XOp>(op) && !isa<YOp>(op) && !isa<ZOp>(op)) {
      return failure();
    }

    // op needs to be in front of a Hadamard gate
    auto hadamardGate = dyn_cast<HOp>(*op->getUsers().begin());

    if (!hadamardGate) {
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
 * measurement, we flip the CNOT and apply a Hadamard gate to the incoming and
 * outcoming qubits. As H * H = id, the measurement is then the direct successor
 * of a CNOT control, which is beneficial for the qubit reuse routine. In that
 * case, measurement lifting (a routine of qubit reuse) can remove the
 * multi-qubit gate by lifting the measurement in front of the control and
 * changing the qubit-controlled Pauli-X to a classically-controlled Pauli-X.
 *
 * The procedure also works if there are additional controls. Only the target
 * and control involved in the transformation get Hadamard gates assigned.
 * The involved ctrl to be flipped with the target is chosen randomly.
 */
struct LiftHadamardAboveCNOTPattern final : OpRewritePattern<MeasureOp> {

  explicit LiftHadamardAboveCNOTPattern(MLIRContext* context)
      : OpRewritePattern(context) {}

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
    auto hadamardGate = dyn_cast<HOp>(predecessor);
    if (!hadamardGate) {
      return failure();
    }

    // The Hadamard gate must be successor of the target of a CNOT
    const auto inQubitHadamard = hadamardGate.getInputQubit(0);
    predecessor = inQubitHadamard.getDefiningOp();
    auto cnotGate = dyn_cast<CtrlOp>(predecessor);
    if (!cnotGate) {
      return failure();
    }
    if (auto innerUnitary =
            utils::getSoleBodyUnitary<UnitaryOpInterface>(*cnotGate.getBody());
        !innerUnitary || !isa<XOp>(innerUnitary.getOperation()) ||
        cnotGate.getOutputTarget(0) != inQubitHadamard) {
      return failure();
    }

    // Find a control qubit not followed by a measurement.
    // If there is no such control, the transformation cannot be applied.
    unsigned int controlIndex = 0;
    for (unsigned int i = 0; i < cnotGate.getNumControls(); i++) {
      if (isa<MeasureOp>(*cnotGate.getOutputControl(i).getUsers().begin())) {
        if (i == cnotGate.getNumControls() - 1) {
          return failure();
        }
      } else {
        controlIndex = i;
        break;
      }
    }

    // Save all SSA values that will be needed after in-place modifications.
    const Value origTgtIn = cnotGate.getInputTarget(0);
    const Value origCtrlIn = cnotGate.getInputControl(controlIndex);
    const Value origCtrlOut = cnotGate.getOutputControl(controlIndex);

    // Add Hadamard gates before the CNOT.
    rewriter.setInsertionPoint(cnotGate);
    auto h1 = HOp::create(rewriter, cnotGate->getLoc(), origTgtIn);
    auto h2 = HOp::create(rewriter, cnotGate->getLoc(), origCtrlIn);

    // Rewire the CNOT operands in-place so that the roles are swapped
    rewriter.modifyOpInPlace(cnotGate, [&]() {
      cnotGate->setOperand(controlIndex, h1.getOutputTarget(0));
      cnotGate->setOperand(cnotGate.getNumControls(), h2.getOutputTarget(0));
    });

    // Move hadamardGate immediately after the CNOT so that it dominates any
    // downstream user of origCtrlOut that might appear before hadamardGate.
    rewriter.moveOpAfter(hadamardGate, cnotGate);

    // Feed the MeasureOp directly from origCtrlOut (the new control output).
    rewriter.modifyOpInPlace(op, [&]() { op->setOperand(0, origCtrlOut); });

    // Every other downstream user of origCtrlOut except for the MeasureOp
    // should now see hadamardGate's output.
    rewriter.replaceAllUsesExcept(origCtrlOut, hadamardGate.getOutputTarget(0),
                                  op);

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
