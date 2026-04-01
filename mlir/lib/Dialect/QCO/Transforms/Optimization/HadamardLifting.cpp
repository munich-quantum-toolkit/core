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
 * @brief Lifts controlled Hadamard gates in front of controlled Pauli gates.
 *
 * This pattern lifts controlled Hadamard gates in front of controlled Pauli
 * gates. This can be done if the output target of the Pauli gate is the input
 * target of the Hadamard gate. Also, all control output qubits need to be the
 * input control qubits of the Hadamard gates. If the Pauli gate is a Z gate,
 * the target and control qubits can be changed upfront in order to be able to
 * lift it.
 */
struct LiftCtrldHadamardsAboveCtrldPauliGatesPattern final
    : mlir::OpRewritePattern<CtrlOp> {

  explicit LiftCtrldHadamardsAboveCtrldPauliGatesPattern(
      mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

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
   * @brief This method swaps a controlled gate with is succeeding controlled
   * Hadamard gate, if applicable.
   *
   * This method swaps a controlled gate with its succeeding controlled
   * Hadamard gate. This is only done if there is a simple commutation rule to
   * do so. Currently implemented:
   * - X - H - = - H - Z -
   * - Y - H - = - H - Y -
   * - Z - H - = - H - X -
   *
   * @param ctrlGate The controlled unitary gate.
   * @param ctrlHadamardGate The controlled hadamard gate.
   * @param rewriter The used rewriter.
   * @return success() if circuit was changed, failure() otherwise
   */
  static mlir::LogicalResult
  swapGateWithHadamardControlled(CtrlOp ctrlGate, CtrlOp ctrlHadamardGate,
                                 mlir::PatternRewriter& rewriter) {
    auto gate = ctrlGate.getBodyUnitary();
    auto hadamardGate = ctrlHadamardGate.getBodyUnitary();
    const auto gateName = gate->getName().stripDialect().str();

    if (gateName == "x" || gateName == "y" || gateName == "z") {
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
    return failure();
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
   * @brief Lifts controlled Hadamard gates in front of controlled Pauli gates.
   *
   * This pattern lifts controlled Hadamard gates in front of controlled Pauli
   * gates. This can be done if the output target of the Pauli gate is the input
   * target of the Hadamard gate. Also, all control output qubits need to be the
   * input control qubits of the Hadamard gates. If the Pauli gate is a Z gate,
   * the target and control qubits can be changed upfront in order to be able to
   * lift it.
   *
   * @param op The controlled operation to match (only controlled Pauli gates
   * trigger the rewrite)
   * @param rewriter Pattern rewriter for applying transformations
   * @return success() if circuit was changed, failure() otherwise
   */
  mlir::LogicalResult
  matchAndRewrite(CtrlOp op, mlir::PatternRewriter& rewriter) const override {
    // op needs to be a controlled Pauli gate
    std::string opName = op.getBodyUnitary()->getName().stripDialect().str();
    if (opName != "x" && opName != "y" && opName != "z") {
      return failure();
    }

    // op needs to be in front of a controlled Hadamard gate
    const auto& users = op->getUsers();
    if (users.empty()) {
      return failure();
    }
    auto user = *users.begin();
    auto userName = user->getName().stripDialect().str();
    if (userName != "ctrl") {
      return failure();
    }
    auto ctrldUser = mlir::dyn_cast<CtrlOp>(user);
    if (ctrldUser.getBodyUnitary()->getName().stripDialect().str() != "h") {
      return failure();
    }

    if (op.getNumTargets() != 1 || ctrldUser.getNumTargets() != 1 ||
        op.getOutputTarget(0) != ctrldUser.getInputTarget(0)) {
      return failure();
    }

    if (!areControlsControlledBySameQubits(op, ctrldUser)) {
      return failure();
    }

    return swapGateWithHadamardControlled(op, ctrldUser, rewriter);
  }
};

/**
 * @brief This pattern is responsible for lifting uncontrolled Hadamard gates
 * above uncontrolled Pauli gates.
 */
struct LiftHadamardsAbovePauliGatesPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  explicit LiftHadamardsAbovePauliGatesPattern(MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  /**
   * @brief This method swaps a gate with is succeeding Hadamard gate, if
   * applicable.
   *
   * This method swaps an uncontrolled gate with its succeeding uncontrolled
   * Hadamard gate. This is only done if there is a simple commutation rule to
   * do so. Currently implemented:
   * - X - H - = - H - Z -
   * - Y - H - = - H - Y -
   * - Z - H - = - H - X -
   *
   * @param gate The unitary gate.
   * @param hadamardGate The hadamard gate.
   * @param rewriter The used rewriter.
   */
  static mlir::LogicalResult
  swapGateWithHadamardUncontrolled(UnitaryOpInterface gate,
                                   UnitaryOpInterface hadamardGate,
                                   mlir::PatternRewriter& rewriter) {
    const auto gateName = gate->getName().stripDialect().str();

    if (gateName == "x" || gateName == "y" || gateName == "z") {
      auto newHadamardGate = rewriter.replaceOpWithNewOp<HOp>(
          gate, gate.getOutputQubit(0).getType(), gate.getInputQubit(0));
      if (gateName == "x") {
        auto newPauliGate = rewriter.replaceOpWithNewOp<ZOp>(
            hadamardGate, hadamardGate.getOutputQubit(0).getType(),
            hadamardGate.getInputQubit(0));
        rewriter.moveOpBefore(newHadamardGate, newPauliGate);
      } else if (gateName == "z") {
        auto newPauliGate = rewriter.replaceOpWithNewOp<XOp>(
            hadamardGate, hadamardGate.getOutputQubit(0).getType(),
            hadamardGate.getInputQubit(0));
        rewriter.moveOpBefore(newHadamardGate, newPauliGate);
      } else {
        auto newPauliGate = rewriter.replaceOpWithNewOp<YOp>(
            hadamardGate, hadamardGate.getOutputQubit(0).getType(),
            hadamardGate.getInputQubit(0));
        rewriter.moveOpBefore(newHadamardGate, newPauliGate);
      }
      return success();
    }
    return failure();
  }

  /**
   * @brief Lifts uncontrolled Hadamard gates in front of uncontrolled Pauli
   * gates.
   *
   * @param op The operation to match (only Pauli gates trigger the rewrite)
   * @param rewriter Pattern rewriter for applying transformations
   * @return success() if circuit was changed, failure() otherwise
   */
  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    // op needs to be a Pauli gate
    std::string opName = op->getName().stripDialect().str();
    if (opName != "x" && opName != "y" && opName != "z") {
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
      return failure();
    }

    auto hadamardGate = mlir::dyn_cast<UnitaryOpInterface>(user);

    if (op.getNumControls() > 0 || hadamardGate.getNumControls() > 0 ||
        op.getNumTargets() != 1 || hadamardGate.getNumTargets() != 1 ||
        op.getOutputTarget(0) != hadamardGate.getInputTarget(0)) {
      return failure();
    }

    return swapGateWithHadamardUncontrolled(op, hadamardGate, rewriter);
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
    patterns.add<LiftCtrldHadamardsAboveCtrldPauliGatesPattern>(
        patterns.getContext());
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