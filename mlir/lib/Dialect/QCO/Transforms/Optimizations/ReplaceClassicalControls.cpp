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
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cassert>
#include <cstddef>
#include <optional>
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
 * @return true if the operation is a phase gate, false otherwise
 */
static bool isPhaseGate(Operation* op) {
  return isa<ZOp, SOp, TOp, POp, SdgOp, TdgOp, IdOp, RZOp>(op);
}

static std::pair<SmallVector<Value>, Value>
applyControlledPhase(PatternRewriter& rewriter, Location loc,
                     ValueRange controls, Value target, Value theta) {
  if (controls.empty()) {
    return {{}, POp::create(rewriter, loc, target, theta).getOutputQubit(0)};
  }
  auto phase = CtrlOp::create(
      rewriter, loc, controls, target, [&](Value innerTarget) -> Value {
        return POp::create(rewriter, loc, innerTarget, theta).getOutputQubit(0);
      });
  return {SmallVector<Value>(phase.getOutputControls()),
          phase.getOutputTarget(0)};
}

static SmallVector<Value> applyConjunctionPhase(PatternRewriter& rewriter,
                                                Location loc,
                                                ValueRange controls,
                                                Value theta) {
  assert(!controls.empty());
  auto [prefix, last] = applyControlledPhase(
      rewriter, loc, controls.drop_back(), controls.back(), theta);
  prefix.push_back(last);
  return prefix;
}

/**
 * @brief Rewrite controlled RZ into a symmetric controlled phase plus its
 * phase correction.
 *
 * For controls C and selected control c, uses
 * C^nRZ(theta) = C^(n-1)P_c(-theta/2) C^nP_target(theta). The latter phase
 * operation is symmetric, so c and the original target may subsequently be
 * swapped.
 *
 * @return The updated controls, including the corrected selected control.
 */
static SmallVector<Value>
rewriteRZForControlTargetSwap(CtrlOp ctrlOp, RZOp rzOp,
                              const size_t selectedControlIndex,
                              PatternRewriter& rewriter) {
  utils::hoistSupportingOpsBefore(*ctrlOp.getBody(), rzOp, ctrlOp, rewriter);
  const Value theta = rzOp.getTheta();

  rewriter.setInsertionPoint(rzOp);
  rewriter.replaceOpWithNewOp<POp>(rzOp, rzOp.getInputTarget(0), theta);

  rewriter.setInsertionPoint(ctrlOp);
  const auto minusHalf =
      utils::constantFromScalar(rewriter, ctrlOp.getLoc(), -0.5);
  const Value correction =
      arith::MulFOp::create(rewriter, ctrlOp.getLoc(), theta, minusHalf);

  SmallVector<Value> controls(ctrlOp.getControlsIn());
  SmallVector<Value> otherControls;
  otherControls.reserve(controls.size() - 1U);
  for (auto [index, control] : llvm::enumerate(controls)) {
    if (index != selectedControlIndex) {
      otherControls.push_back(control);
    }
  }
  Value selectedControl = controls[selectedControlIndex];
  std::tie(otherControls, selectedControl) = applyControlledPhase(
      rewriter, ctrlOp.getLoc(), otherControls, selectedControl, correction);

  size_t otherIndex = 0;
  for (auto [index, control] : llvm::enumerate(controls)) {
    if (index == selectedControlIndex) {
      control = selectedControl;
    } else {
      control = otherControls[otherIndex++];
    }
  }
  return controls;
}

/**
 * @brief Replace a controlled RZZ with one measured target by classical
 * control flow.
 *
 * For the conjunction d of all controls, measured target a, and remaining
 * target b, the phase polynomial identity is
 *
 * C^nRZZ(theta) = C^(n-1)P_d(-theta/2) C^nP_b(theta)
 *                   C_a[C^(n-1)P_d(theta) C^nP_b(-2 theta)].
 */
static LogicalResult tryReplaceMeasuredRZZTarget(CtrlOp op, RZZOp rzzOp,
                                                 PatternRewriter& rewriter) {
  if (op.getNumControls() == 0 || op.getNumTargets() != 2) {
    return failure();
  }

  std::optional<size_t> measuredTargetIndex;
  Value condition;
  for (auto [index, target] : llvm::enumerate(op.getTargetsIn())) {
    if (Value outcome = getPredecessorMeasurementOutcome(target)) {
      if (measuredTargetIndex) {
        return failure();
      }
      measuredTargetIndex = index;
      condition = outcome;
    }
  }
  if (!measuredTargetIndex) {
    return failure();
  }

  const size_t otherTargetIndex = 1U - *measuredTargetIndex;
  const Value measuredTarget = op.getInputTarget(*measuredTargetIndex);
  SmallVector<Value> controls(op.getControlsIn());
  Value otherTarget = op.getInputTarget(otherTargetIndex);

  utils::hoistSupportingOpsBefore(*op.getBody(), rzzOp, op, rewriter);
  const Value theta = rzzOp.getTheta();
  rewriter.setInsertionPoint(op);
  const Value minusHalf =
      utils::constantFromScalar(rewriter, op.getLoc(), -0.5);
  const Value minusTwo = utils::constantFromScalar(rewriter, op.getLoc(), -2.0);
  const Value negativeHalfTheta =
      arith::MulFOp::create(rewriter, op.getLoc(), theta, minusHalf);
  const Value negativeDoubleTheta =
      arith::MulFOp::create(rewriter, op.getLoc(), theta, minusTwo);

  controls =
      applyConjunctionPhase(rewriter, op.getLoc(), controls, negativeHalfTheta);
  std::tie(controls, otherTarget) =
      applyControlledPhase(rewriter, op.getLoc(), controls, otherTarget, theta);

  SmallVector<Value> ifOperands(controls);
  ifOperands.push_back(otherTarget);
  auto ifOp = IfOp::create(
      rewriter, op.getLoc(), condition, ifOperands,
      [&](ValueRange qubits) -> SmallVector<Value> {
        SmallVector<Value> conditionalControls(qubits.drop_back());
        conditionalControls = applyConjunctionPhase(rewriter, op.getLoc(),
                                                    conditionalControls, theta);
        Value conditionalTarget;
        std::tie(conditionalControls, conditionalTarget) =
            applyControlledPhase(rewriter, op.getLoc(), conditionalControls,
                                 qubits.back(), negativeDoubleTheta);
        conditionalControls.push_back(conditionalTarget);
        return conditionalControls;
      });

  SmallVector<Value> replacements(ifOp.getLinearResults().drop_back());
  replacements.resize(op.getNumQubits());
  replacements[op.getNumControls() + *measuredTargetIndex] = measuredTarget;
  replacements[op.getNumControls() + otherTargetIndex] =
      ifOp.getLinearResults().back();
  rewriter.replaceOp(op, replacements);
  return success();
}

/**
 * @brief For a phase gate whose target has a predecessor measurement, swaps the
 * target with an eligible control.
 * @param op The control operation containing the phase gate
 * @param rewriter The pattern rewriter used to perform the transformation
 */
static void trySwapControlAndTargetOfPhaseGate(CtrlOp op,
                                               UnitaryOpInterface unitary,
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

    Value swappedTarget = control;
    SmallVector<Value> updatedControls(op.getControlsIn());
    if (auto rzOp = dyn_cast<RZOp>(unitary.getOperation())) {
      updatedControls =
          rewriteRZForControlTargetSwap(op, rzOp, controlIndex, rewriter);
      swappedTarget = updatedControls[controlIndex];
    }

    Value controlOut = op.getControlsOut()[controlIndex];
    Value targetOut = op.getTargetsOut()[0];

    rewriter.modifyOpInPlace(op, [&]() {
      for (auto [index, updatedControl] : llvm::enumerate(updatedControls)) {
        op.getControlsInMutable()[index].set(updatedControl);
      }
      op.getTargetsInMutable()[0].set(swappedTarget);
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

    if (auto unitary =
            utils::getSoleBodyUnitary<UnitaryOpInterface>(*ctrlOp.getBody());
        unitary) {
      if (auto rzzOp = dyn_cast<RZZOp>(unitary.getOperation());
          rzzOp &&
          succeeded(tryReplaceMeasuredRZZTarget(ctrlOp, rzzOp, rewriter))) {
        return success();
      }
      if (isPhaseGate(unitary.getOperation())) {
        trySwapControlAndTargetOfPhaseGate(ctrlOp, unitary, rewriter);
        rewriter.setInsertionPointAfter(ctrlOp);
      }
    }

    ValueRange controlsIn = ctrlOp.getControlsIn();
    ValueRange controlResults = ctrlOp.getControlsOut();

    SmallVector<Value> ifOperands;
    ifOperands.reserve(ctrlOp.getNumQubits());
    SmallVector<Value> oldOutputs;
    oldOutputs.reserve(ctrlOp->getNumResults());
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
        ifOperands.push_back(control);
        oldOutputs.push_back(oldOutput);
      }
    }

    if (!condition) {
      return failure();
    }

    const auto numRemaining = ifOperands.size();
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

    rewriter.replaceAllUsesWith(oldOutputs, ifOp.getLinearResults());
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
