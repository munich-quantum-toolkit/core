/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/Utils/Modifiers.h"
#include "mlir/Dialect/MQT/Utils/Parameters.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <optional>
#include <tuple>
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
 * @brief Checks if the given operation applies a phase only to the target's
 * one state.
 * @param op The operation to check
 * @return true if the operation is a phase gate, false otherwise
 */
static bool isPhaseGate(Operation* op) {
  return isa<ZOp, SOp, TOp, POp, SdgOp, TdgOp, IdOp>(op);
}

/**
 * @brief Select a scalar based on @p condition and multiply @p theta by it.
 */
static Value selectScaledAngle(PatternRewriter& rewriter, Location loc,
                               Value theta, Value condition,
                               const double trueScale,
                               const double falseScale) {
  Value trueValue = mqt::constantFromScalar(rewriter, loc, trueScale);
  Value falseValue = mqt::constantFromScalar(rewriter, loc, falseScale);
  Value scale =
      arith::SelectOp::create(rewriter, loc, condition, trueValue, falseValue);
  return arith::MulFOp::create(rewriter, loc, theta, scale);
}

/**
 * @brief Apply a phase gate to @p target, controlled by @p controls.
 * @return A pair containing the updated controls in their input order and the
 * updated target.
 */
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

/**
 * @brief Apply an RZ gate to @p target, controlled by @p controls.
 * @return A pair containing the updated controls in their input order and the
 * updated target.
 */
static std::pair<SmallVector<Value>, Value>
applyControlledRZ(PatternRewriter& rewriter, Location loc, ValueRange controls,
                  Value target, Value theta) {
  if (controls.empty()) {
    return {{}, RZOp::create(rewriter, loc, target, theta).getOutputQubit(0)};
  }
  auto rz = CtrlOp::create(
      rewriter, loc, controls, target, [&](Value innerTarget) -> Value {
        return RZOp::create(rewriter, loc, innerTarget, theta)
            .getOutputQubit(0);
      });
  return {SmallVector<Value>(rz.getOutputControls()), rz.getOutputTarget(0)};
}

/**
 * @brief Apply a phase to the conjunction of @p controls, using the last
 * control as the phase target.
 * @return The updated controls in their input order.
 */
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
 * @brief Check whether all qubits are direct measurement results.
 */
static bool areAllMeasured(ValueRange qubits) {
  return llvm::all_of(qubits, [](Value qubit) {
    return static_cast<bool>(getPredecessorMeasurementOutcome(qubit));
  });
}

/**
 * @brief Map each control target result to the corresponding input target.
 * @return The input-target index for every target result, or @c std::nullopt
 * if the body does not directly yield all results of @p rzzOp.
 */
static std::optional<SmallVector<size_t>> getRZZTargetResultOrder(CtrlOp ctrlOp,
                                                                  RZZOp rzzOp) {
  SmallVector<size_t> resultOrder;
  resultOrder.reserve(ctrlOp.getNumTargets());
  auto yieldOp = cast<YieldOp>(ctrlOp.getBody()->getTerminator());
  for (Value yielded : yieldOp.getOperands()) {
    auto result = dyn_cast<OpResult>(yielded);
    if (!result || result.getOwner() != rzzOp.getOperation()) {
      return std::nullopt;
    }
    auto input =
        dyn_cast<BlockArgument>(rzzOp.getInputTarget(result.getResultNumber()));
    if (!input || input.getOwner() != ctrlOp.getBody() ||
        input.getArgNumber() >= ctrlOp.getNumTargets()) {
      return std::nullopt;
    }
    resultOrder.push_back(input.getArgNumber());
  }
  return resultOrder;
}

/**
 * @brief Replace @p ctrlOp while preserving its body-yield target order.
 */
static void replaceRZZCtrlOp(CtrlOp ctrlOp, ArrayRef<size_t> targetResultOrder,
                             ValueRange controlsByInput,
                             ValueRange targetsByInput,
                             PatternRewriter& rewriter) {
  SmallVector<Value> replacements(controlsByInput);
  replacements.reserve(ctrlOp.getNumQubits());
  for (const size_t inputIndex : targetResultOrder) {
    replacements.push_back(targetsByInput[inputIndex]);
  }
  rewriter.replaceOp(ctrlOp, replacements);
}

/**
 * @brief Replace a controlled RZ whose target has already been measured.
 *
 * On a measured target, RZ contributes only an outcome-dependent phase to the
 * conjunction of the controls. If every participating qubit has been measured,
 * that phase is unobservable and the operation is removed.
 */
static LogicalResult tryReplaceMeasuredRZTarget(CtrlOp op, RZOp rzOp,
                                                PatternRewriter& rewriter) {
  if (op.getNumTargets() != 1) {
    return failure();
  }
  Value outcome = getPredecessorMeasurementOutcome(op.getInputTarget(0));
  if (!outcome) {
    return failure();
  }
  if (areAllMeasured(op.getControlsIn())) {
    rewriter.replaceOp(op, op.getInputQubits());
    return success();
  }

  mqt::hoistSupportingOpsBefore(*op.getBody(), rzOp, op, rewriter);
  rewriter.setInsertionPoint(op);
  Value phase = selectScaledAngle(rewriter, op.getLoc(), rzOp.getTheta(),
                                  outcome, 0.5, -0.5);
  SmallVector<Value> replacements =
      applyConjunctionPhase(rewriter, op.getLoc(), op.getControlsIn(), phase);
  replacements.push_back(op.getInputTarget(0));
  rewriter.replaceOp(op, replacements);
  return success();
}

/**
 * @brief Replace a controlled RZZ with one or two measured targets.
 *
 * Fixing one measured target reduces RZZ to an outcome-dependent RZ on the
 * other target. Fixing both targets leaves only an outcome-dependent phase on
 * the conjunction of the controls. The operation is removed when all of those
 * controls have also been measured.
 */
static LogicalResult tryReplaceMeasuredRZZTarget(CtrlOp op, RZZOp rzzOp,
                                                 PatternRewriter& rewriter) {
  if (op.getNumTargets() != 2) {
    return failure();
  }

  const auto targetResultOrder = getRZZTargetResultOrder(op, rzzOp);
  if (!targetResultOrder) {
    return failure();
  }

  std::array<Value, 2> targetOutcomes;
  for (auto [index, target] : llvm::enumerate(op.getTargetsIn())) {
    targetOutcomes[index] = getPredecessorMeasurementOutcome(target);
  }
  const bool bothTargetsMeasured = targetOutcomes[0] && targetOutcomes[1];
  if (!targetOutcomes[0] && !targetOutcomes[1]) {
    return failure();
  }
  if (bothTargetsMeasured && areAllMeasured(op.getControlsIn())) {
    replaceRZZCtrlOp(op, *targetResultOrder, op.getControlsIn(),
                     op.getTargetsIn(), rewriter);
    return success();
  }

  mqt::hoistSupportingOpsBefore(*op.getBody(), rzzOp, op, rewriter);
  rewriter.setInsertionPoint(op);
  SmallVector<Value> controls(op.getControlsIn());
  SmallVector<Value> targets(op.getTargetsIn());

  if (bothTargetsMeasured) {
    Value parity = arith::XOrIOp::create(rewriter, op.getLoc(),
                                         targetOutcomes[0], targetOutcomes[1]);
    Value phase = selectScaledAngle(rewriter, op.getLoc(), rzzOp.getTheta(),
                                    parity, 0.5, -0.5);
    controls = applyConjunctionPhase(rewriter, op.getLoc(), controls, phase);
  } else {
    const size_t measuredTarget = targetOutcomes[0] ? 0U : 1U;
    const size_t otherTarget = 1U - measuredTarget;
    Value angle = selectScaledAngle(rewriter, op.getLoc(), rzzOp.getTheta(),
                                    targetOutcomes[measuredTarget], -1.0, 1.0);
    std::tie(controls, targets[otherTarget]) = applyControlledRZ(
        rewriter, op.getLoc(), controls, targets[otherTarget], angle);
  }

  replaceRZZCtrlOp(op, *targetResultOrder, controls, targets, rewriter);
  return success();
}

/**
 * @brief For a phase gate whose target has a predecessor measurement, swaps the
 * target with an eligible control.
 * @param op The control operation containing the phase gate
 * @param rewriter The pattern rewriter used to perform the transformation
 */
static void trySwapControlAndTargetOfPhaseGate(CtrlOp op,
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

    if (auto unitary =
            mqt::getSoleBodyUnitary<UnitaryOpInterface>(*ctrlOp.getBody());
        unitary) {
      if (auto rzOp = dyn_cast<RZOp>(unitary.getOperation());
          rzOp &&
          succeeded(tryReplaceMeasuredRZTarget(ctrlOp, rzOp, rewriter))) {
        return success();
      }
      if (auto rzzOp = dyn_cast<RZZOp>(unitary.getOperation());
          rzzOp &&
          succeeded(tryReplaceMeasuredRZZTarget(ctrlOp, rzzOp, rewriter))) {
        return success();
      }
      if (isPhaseGate(unitary.getOperation())) {
        trySwapControlAndTargetOfPhaseGate(ctrlOp, rewriter);
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
        if (!condition) {
          condition = outcome;
        } else {
          condition = arith::AndIOp::create(rewriter, ctrlOp.getLoc(),
                                            condition, outcome);
        }
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
    auto op = getOperation();
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
