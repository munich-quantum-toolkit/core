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
   * @brief Lifts Hadamard gates in front of Pauli gates.
   *
   * @param op The operation to match (only Pauli gates trigger the rewrite)
   * @param rewriter Pattern rewriter for applying transformations
   * @return success() if circuit was changed, failure() otherwise
   */
  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    return failure();
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