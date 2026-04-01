/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "../../../../../../include/mqt-core/ir/operations/OpType.hpp"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
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
 * @brief This pattern changes the target of a controlled Pauli Z gate if a
 * controlled hadamard gate is it successor.
 * If all out qubits of Pauli Z are equal to all in qubits of Hadamard, we can
 * commute the gates and change Pauli Z to X. This is only possible if Hadamard
 * and Pauli act on the same qubit as target. If the target of the Pauli gate is
 * a ctrl at the hadamard and vice versa, we can change the target of Pauli Z to
 * the Hadamard's. This is done in this pattern.
 */
struct AdaptCtrldPauliZToLiftingPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryOpInterface> {

  explicit AdaptCtrldPauliZToLiftingPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  /**
   * @brief Changes the target of a controlled Pauli Z gate if a
   * controlled hadamard gate is it successor.
   *
   * @param op The operation to match (only Pauli gates trigger the rewrite)
   * @param rewriter Pattern rewriter for applying transformations
   * @return success() if circuit was changed, failure() otherwise
   */
  mlir::LogicalResult
  matchAndRewrite(UnitaryOpInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    return failure();
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
    bool result = true;
    result &= range1.size() == range2.size();
    for (auto element : range1) {
      result &=
          std::find(range2.begin(), range2.end(), element) != range2.end();
    }
    return result;
  }

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
    std::vector<Value> controlOutputsFirstGate(
        firstCtrl.getControlsOut().begin(), firstCtrl.getControlsOut().end());
    std::vector<Value> controlInputsSecondGate(
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
struct LiftHadamardAboveCNOTPattern final : mlir::OpRewritePattern<MeasureOp> {

  explicit LiftHadamardAboveCNOTPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  /**
   * @brief This pattern remove an H gate between a CNOT and a measurement.
   *
   * @param op The operation to match (only uncontrolled Hadamard gates trigger
   * the rewrite)
   * @param rewriter Pattern rewriter for applying transformations
   * @return success() if circuit was changed, failure() otherwise
   */
  mlir::LogicalResult
  matchAndRewrite(MeasureOp op,
                  mlir::PatternRewriter& rewriter) const override {
    return mlir::failure();
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
    auto op = getOperation();
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