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
   * @brief This method checks if two gates are connected by exactly the same
   * target and ctrl qubits.
   *
   * This method checks if the output target/ctrl qubits of the first gate are
   * exactly the input target/ctrl qubits of the second gate. There must be no
   * qubit that is only used by one of the gates.
   *
   * @param firstGate The first unitary gate.
   * @param secondGate The second unitary gate.
   */
  static bool
  areGatesConnectedExactlyBySameQubits(UnitaryOpInterface firstGate,
                                       UnitaryOpInterface secondGate) {
    if (firstGate.getNumTargets() != secondGate.getNumTargets() ||
        firstGate.getNumControls() != secondGate.getNumControls()) {
      return false;
    }
    std::vector<Value> targetOutputsFirstGate;
    std::vector<Value> controlOutputsFirstGate;
    std::vector<Value> targetInputsSecondGate;
    std::vector<Value> controlInputsSecondGate;
    for (size_t i = 0; i < firstGate.getNumTargets(); i++) {
      targetOutputsFirstGate.push_back(firstGate.getOutputTarget(i));
      targetInputsSecondGate.push_back(secondGate.getInputTarget(i));
    }
    for (size_t i = 0; i < firstGate.getNumControls(); i++) {
      targetOutputsFirstGate.push_back(firstGate.getOutputControl(i));
      targetInputsSecondGate.push_back(secondGate.getInputControl(i));
    }

    bool result = true;
    result &= containRangesOfSameElements(targetOutputsFirstGate,
                                          targetInputsSecondGate);
    result &= containRangesOfSameElements(controlOutputsFirstGate,
                                          controlInputsSecondGate);
    return result;
  }

  /**
   * @brief This method swaps a gate with is succeeding hadamard gate, if
   * applicable.
   *
   * This method swaps a gate with its suceeding hadamard gate. This is only
   * done if there is a simple commutation rule to do so.
   * Currently implemented:
   * - X - H - = - H - Z -
   * - Y - H - = - H - Y -
   * - Z - H - = - H - X -
   *
   * @param gate The unitary gate.
   * @param hadamardGate The hadamard gate.
   * @param rewriter The used rewriter.
   */
  static mlir::LogicalResult
  swapGateWithHadamard(UnitaryOpInterface gate, UnitaryOpInterface hadamardGate,
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
    if (opName != "x" && opName != "y" && opName != "z") {
      return failure();
    }

    // op needs to be in front of a hadamard gate
    const auto& users = op->getUsers();
    if (users.empty()) {
      return failure();
    }
    auto user = *users.begin();
    if (user->getName().stripDialect().str() != "h") {
      return failure();
    }

    auto hadamardGate = mlir::dyn_cast<UnitaryOpInterface>(user);

    if (!areGatesConnectedExactlyBySameQubits(op, hadamardGate)) {
      return failure();
    }

    return swapGateWithHadamard(op, hadamardGate, rewriter);
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