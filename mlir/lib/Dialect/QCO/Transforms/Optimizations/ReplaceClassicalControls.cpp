/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

//
// Created by damian on 5/13/26.
//

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_REPLACECLASSICALCONTROLS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

/**
 * @brief Retrieves the measurement outcome that directly precedes the given
 * qubit, if it exists.
 * @param qubit The qubit for which to find the predecessor measurement outcome
 * @return The measurement outcome if a predecessor measurement exists, nullptr
 * otherwise
 */
static Value getPredecessorMeasurementOutcome(Value qubit) {
  auto* definingOp = qubit.getDefiningOp();
  if (auto measureOp = dyn_cast_or_null<MeasureOp>(definingOp)) {
    return measureOp.getResult();
  }
  return nullptr;
}

/**
 * @brief Checks if the given operation is a diagonal gate, i.e., it only
 * applies a phase to the target qubit(s) and does not change their state.
 * @param op The operation to check
 * @return true if the operation is a diagonal gate, false otherwise
 */
static bool isDiagonal(Operation* op) {
  if (auto i = dyn_cast<InvOp>(op)) {
    return isDiagonal(i.getBodyUnitary());
  }
  return isa<ZOp, SOp, TOp, POp, SdgOp, TdgOp, IdOp>(op);
}

/**
 * @brief For a diagonal gate with a control that has a predecessor measurement,
 * swaps the control with the target.
 * @param op The control operation containing the diagonal gate
 * @param rewriter The pattern rewriter used to perform the transformation
 */
static void trySwapControlsOfDiagonalGate(CtrlOp op,
                                          mlir::PatternRewriter& rewriter) {
  assert(op.getBodyUnitary().getNumQubits() == 1 &&
         "Only single-qubit gates can be swapped around controls");
  auto target = op.getTargetsIn()[0];
  auto predecessorOutcome = getPredecessorMeasurementOutcome(target);
  if (!predecessorOutcome) {
    // No advantage gained from swapping.
    return;
  }
  for (auto control : op.getControlsIn()) {
    auto controlOutcome = getPredecessorMeasurementOutcome(control);
    if (controlOutcome) {
      continue;
    }
    rewriter.replaceAllUsesWith(control, target);
    rewriter.modifyOpInPlace(
        op, [&]() { op.getTargetsInMutable()[0].set(control); });
    auto dummyTarget = AllocOp::create(rewriter, op->getLoc());
    rewriter.replaceAllUsesWith(op.getOutputForInput(target), dummyTarget);
    rewriter.replaceAllUsesWith(op.getOutputForInput(control),
                                op.getOutputForInput(target));
    rewriter.replaceAllUsesWith(dummyTarget, op.getOutputForInput(control));
    rewriter.eraseOp(dummyTarget);
    break;
  }
}

namespace {
/**
 * @brief This pattern is responsible for replacing controls after measurements
 * with `if` constructs.
 */
struct ReplaceBasisStateControlsWithIfPattern final
    : mlir::OpRewritePattern<CtrlOp> {

  explicit ReplaceBasisStateControlsWithIfPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(CtrlOp op, mlir::PatternRewriter& rewriter) const override {
    if (isDiagonal(op.getBodyUnitary())) {
      trySwapControlsOfDiagonalGate(op, rewriter);
    }

    SmallVector<std::pair<mlir::Value, mlir::Value>> toReplace;
    SmallVector<mlir::Value> toKeep;

    for (const auto& operand : op.getControlsIn()) {
      auto outcome = getPredecessorMeasurementOutcome(operand);
      if (outcome) {
        toReplace.emplace_back(operand, outcome);
      } else {
        toKeep.push_back(operand);
      }
    }

    if (toReplace.empty()) {
      return mlir::failure();
    }

    auto condition = std::accumulate(
        std::next(toReplace.begin()), toReplace.end(),
        toReplace.begin()->second, [&](const auto& acc, const auto& pair) {
          auto conjunction =
              arith::AndIOp::create(rewriter, op.getLoc(), acc, pair.second);
          return conjunction.getResult();
        });

    auto allQubits = toKeep;
    llvm::append_range(allQubits, op.getTargetsIn());

    auto ifOp = IfOp::create(
        rewriter, op->getLoc(), condition, allQubits,
        [&](ValueRange qubits) -> SmallVector<Value> {
          auto newControls = qubits.slice(0, toKeep.size());
          auto newTargets =
              qubits.slice(toKeep.size(), qubits.size() - toKeep.size());

          auto newCtrl =
              CtrlOp::create(rewriter, op->getLoc(), newControls, newTargets);
          rewriter.inlineRegionBefore(op.getRegion(), newCtrl.getRegion(),
                                      newCtrl.getRegion().begin());
          return newCtrl.getOutputQubits();
        });

    for (auto replace : toReplace) {
      rewriter.replaceAllUsesWith(op.getOutputForInput(replace.first),
                                  replace.first);
    }
    for (auto [oldInput, result] : llvm::zip(allQubits, ifOp.getResults())) {
      rewriter.replaceAllUsesWith(op.getOutputForInput(oldInput), result);
    }
    rewriter.eraseOp(op);

    return mlir::success();
  }
};

/**
 * @brief Pass replaces controls with `IfOp` operations if the qubits'
 * control values are available classically.
 */
struct ReplaceClassicalControls final
    : impl::ReplaceClassicalControlsBase<ReplaceClassicalControls> {
  using ReplaceClassicalControlsBase::ReplaceClassicalControlsBase;

protected:
  void runOnOperation() override {
    const auto op = getOperation();
    auto* ctx = &getContext();

    // Define the set of patterns to use.
    RewritePatternSet patterns(ctx);
    patterns.add<ReplaceBasisStateControlsWithIfPattern>(patterns.getContext());

    // Apply patterns in an iterative and greedy manner.
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco
