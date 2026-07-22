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
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cassert>
#include <cstddef>
#include <iterator>
#include <numeric>
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
 * @brief Checks if the given operation is a phase gate, i.e., it only
 * applies a phase to the target qubit(s) in the 1 state.
 * @param op The operation to check
 * @return true if the operation is a diagonal gate, false otherwise
 */
static bool isPhaseGate(Operation* op) {
  return isa<ZOp, SOp, TOp, POp, SdgOp, TdgOp, IdOp>(op);
}

/**
 * @brief For a diagonal gate with a control that has a predecessor measurement,
 * swaps the control with the target.
 * @param op The control operation containing the diagonal gate
 * @param rewriter The pattern rewriter used to perform the transformation
 */
static void trySwapControlsOfDiagonalGate(CtrlOp op,
                                          PatternRewriter& rewriter) {
  assert(op.getNumTargets() == 1 &&
         "Only single-qubit gates can be swapped around controls");
  auto target = op.getTargetsIn()[0];
  auto predecessorOutcome = getPredecessorMeasurementOutcome(target);
  if (!predecessorOutcome) {
    // No advantage gained from swapping.
    return;
  }

  size_t controlIndex = 0;
  for (auto control : op.getControlsIn()) {
    auto controlOutcome = getPredecessorMeasurementOutcome(control);
    if (controlOutcome) {
      controlIndex++;
      continue;
    }

    Value controlOut = op.getControlsOut()[controlIndex];
    Value targetOut = op.getTargetsOut()[0];

    rewriter.modifyOpInPlace(op, [&]() {
      op.getTargetsInMutable()[0].set(control);
      op.getControlsInMutable()[controlIndex].set(target);
    });

    // This works because each qubit is only ever used once.
    auto controlUse = controlOut.getUses().begin();
    auto targetUse = targetOut.getUses().begin();
    controlUse->set(targetOut);
    targetUse->set(controlOut);

    break;
  }
}

namespace {
/**
 * @brief This pattern is responsible for replacing controls after measurements
 * with `if` constructs.
 */
struct ReplaceBasisStateControlsWithIfPattern final
    : OpRewritePattern<MeasureOp> {

  explicit ReplaceBasisStateControlsWithIfPattern(MLIRContext* context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(MeasureOp measure,
                                PatternRewriter& rewriter) const override {
    auto ctrlOp = dyn_cast<CtrlOp>(*measure.getQubitOut().getUsers().begin());
    if (!ctrlOp) {
      return failure();
    }
    rewriter.setInsertionPointAfter(ctrlOp);

    if (utils::getSoleBodyUnitary<UnitaryOpInterface>(*ctrlOp.getBody())) {
      trySwapControlsOfDiagonalGate(ctrlOp, rewriter);
    }

    ValueRange controlsIn = ctrlOp.getControlsIn();
    ValueRange controlResults = ctrlOp.getControlsOut();

    SmallVector<Value> remainingControls;
    SmallVector<Value> oldOutputs;
    Value condition;
    for (auto [control, oldOutput] :
         llvm::zip_equal(controlsIn, controlResults)) {
      if (Value outcome = getPredecessorMeasurementOutcome(control)) {
        rewriter.replaceAllUsesWith(oldOutput, control);
        condition = condition ? arith::AndIOp::create(rewriter, ctrlOp.getLoc(),
                                                      condition, outcome)
                                    .getResult()
                              : outcome;
      } else {
        remainingControls.push_back(control);
        oldOutputs.push_back(oldOutput);
      }
    }

    if (!condition) {
      return failure();
    }

    size_t numRemaining = remainingControls.size();
    SmallVector<Value> ifOperands = remainingControls;
    llvm::append_range(ifOperands, ctrlOp.getTargetsIn());
    llvm::append_range(oldOutputs, ctrlOp.getTargetsOut());

    auto ifOp = IfOp::create(
        rewriter, ctrlOp.getLoc(), condition, ifOperands,
        [&](ValueRange qubits) -> SmallVector<Value> {
          auto newCtrl = CtrlOp::create(rewriter, ctrlOp.getLoc(),
                                        qubits.take_front(numRemaining),
                                        qubits.drop_front(numRemaining));
          rewriter.inlineRegionBefore(ctrlOp.getRegion(), newCtrl.getRegion(),
                                      newCtrl.getRegion().begin());
          return newCtrl.getOutputQubits();
        });

    for (auto [oldOutput, result] :
         llvm::zip_equal(oldOutputs, ifOp.getResults())) {
      rewriter.replaceAllUsesWith(oldOutput, result);
    }
    rewriter.eraseOp(ctrlOp);

    return success();
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
