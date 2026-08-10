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
 * @brief Checks if the given operation is an eligible single-target diagonal
 * gate. RZ requires an additional phase correction when exchanging its target
 * with a control.
 * @param op The operation to check
 * @return true if the operation is a phase gate, false otherwise
 */
static bool isPhaseGate(Operation* op) {
  return isa<ZOp, SOp, TOp, POp, SdgOp, TdgOp, IdOp, RZOp>(op);
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
 * @brief Map each control target result to the corresponding input target.
 * @return The input-target index for every target result, or @c std::nullopt
 * if the body does not directly yield all results of @p rzzOp.
 */
static std::optional<SmallVector<size_t>> getRZZTargetResultOrder(CtrlOp ctrlOp,
                                                                  RZZOp rzzOp) {
  SmallVector<size_t> resultOrder;
  resultOrder.reserve(ctrlOp.getNumTargets());
  auto yieldOp = cast<YieldOp>(ctrlOp.getBody()->getTerminator());
  for (const Value yielded : yieldOp.getOperands()) {
    const auto result = dyn_cast<OpResult>(yielded);
    if (!result || result.getOwner() != rzzOp.getOperation()) {
      return std::nullopt;
    }
    const auto input =
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
 * @brief Build the replacement for a controlled RZZ with measured targets.
 * @return Updated quantum controls followed by targets in input order.
 */
static SmallVector<Value>
buildMeasuredRZZReplacement(PatternRewriter& rewriter, Location loc,
                            ValueRange controls, ValueRange targets,
                            const std::array<Value, 2>& targetOutcomes,
                            Value theta) {
  const size_t numMeasuredTargets =
      static_cast<size_t>(llvm::count_if(targetOutcomes, [](Value outcome) {
        return static_cast<bool>(outcome);
      }));
  assert((numMeasuredTargets == 1 || numMeasuredTargets == 2) &&
         "expected at least one measured RZZ target");

  SmallVector<Value> updatedControls(controls);
  SmallVector<Value> updatedTargets(targets);
  if (numMeasuredTargets == 2) {
    if (updatedControls.empty()) {
      return updatedTargets;
    }

    const Value minusHalf = utils::constantFromScalar(rewriter, loc, -0.5);
    const Value negativeHalfTheta =
        arith::MulFOp::create(rewriter, loc, theta, minusHalf);
    updatedControls = applyConjunctionPhase(rewriter, loc, updatedControls,
                                            negativeHalfTheta);
    const Value outcomesDiffer = arith::XOrIOp::create(
        rewriter, loc, targetOutcomes[0], targetOutcomes[1]);
    auto ifOp = IfOp::create(rewriter, loc, outcomesDiffer, updatedControls,
                             [&](ValueRange qubits) -> SmallVector<Value> {
                               return applyConjunctionPhase(rewriter, loc,
                                                            qubits, theta);
                             });
    updatedControls.assign(ifOp.getLinearResults().begin(),
                           ifOp.getLinearResults().end());
  } else {
    const size_t measuredTargetIndex = targetOutcomes[0] ? 0U : 1U;
    const size_t otherTargetIndex = 1U - measuredTargetIndex;
    const Value minusTwo = utils::constantFromScalar(rewriter, loc, -2.0);
    const Value negativeDoubleTheta =
        arith::MulFOp::create(rewriter, loc, theta, minusTwo);

    if (updatedControls.empty()) {
      updatedTargets[otherTargetIndex] =
          RZOp::create(rewriter, loc, updatedTargets[otherTargetIndex], theta)
              .getOutputQubit(0);
      auto ifOp =
          IfOp::create(rewriter, loc, targetOutcomes[measuredTargetIndex],
                       ValueRange{updatedTargets[otherTargetIndex]},
                       [&](ValueRange qubits) -> SmallVector<Value> {
                         return {RZOp::create(rewriter, loc, qubits.front(),
                                              negativeDoubleTheta)
                                     .getOutputQubit(0)};
                       });
      updatedTargets[otherTargetIndex] = ifOp.getLinearResults().front();
    } else {
      const Value minusHalf = utils::constantFromScalar(rewriter, loc, -0.5);
      const Value negativeHalfTheta =
          arith::MulFOp::create(rewriter, loc, theta, minusHalf);
      updatedControls = applyConjunctionPhase(rewriter, loc, updatedControls,
                                              negativeHalfTheta);
      std::tie(updatedControls, updatedTargets[otherTargetIndex]) =
          applyControlledPhase(rewriter, loc, updatedControls,
                               updatedTargets[otherTargetIndex], theta);

      SmallVector<Value> ifOperands(updatedControls);
      ifOperands.push_back(updatedTargets[otherTargetIndex]);
      auto ifOp = IfOp::create(
          rewriter, loc, targetOutcomes[measuredTargetIndex], ifOperands,
          [&](ValueRange qubits) -> SmallVector<Value> {
            SmallVector<Value> conditionalControls(qubits.drop_back());
            conditionalControls = applyConjunctionPhase(
                rewriter, loc, conditionalControls, theta);
            Value conditionalTarget;
            std::tie(conditionalControls, conditionalTarget) =
                applyControlledPhase(rewriter, loc, conditionalControls,
                                     qubits.back(), negativeDoubleTheta);
            conditionalControls.push_back(conditionalTarget);
            return conditionalControls;
          });
      updatedControls.assign(ifOp.getLinearResults().drop_back().begin(),
                             ifOp.getLinearResults().drop_back().end());
      updatedTargets[otherTargetIndex] = ifOp.getLinearResults().back();
    }
  }

  llvm::append_range(updatedControls, updatedTargets);
  return updatedControls;
}

/**
 * @brief Replace a controlled RZZ with measured targets by classical control
 * flow and phase corrections on the remaining quantum qubits.
 *
 * For the conjunction d of the quantum controls, measured target a, and
 * remaining target b, the phase polynomial identity is
 *
 * C^nRZZ(theta) = C^(n-1)P_d(-theta/2) C^nP_b(theta)
 *                   C_a[C^(n-1)P_d(theta) C^nP_b(-2 theta)].
 * Measured controls guard the complete correction classically. If both targets
 * are measured, only phase kickback on quantum controls remains; if there are
 * no quantum controls, the operation can be removed.
 */
static LogicalResult tryReplaceMeasuredRZZTarget(CtrlOp op, RZZOp rzzOp,
                                                 PatternRewriter& rewriter) {
  if (op.getNumControls() == 0 || op.getNumTargets() != 2) {
    return failure();
  }

  const auto targetResultOrder = getRZZTargetResultOrder(op, rzzOp);
  if (!targetResultOrder) {
    return failure();
  }

  std::array<Value, 2> targetOutcomes;
  for (auto [index, target] : llvm::enumerate(op.getTargetsIn())) {
    if (Value outcome = getPredecessorMeasurementOutcome(target)) {
      targetOutcomes[index] = outcome;
    }
  }
  const size_t numMeasuredTargets =
      static_cast<size_t>(llvm::count_if(targetOutcomes, [](Value outcome) {
        return static_cast<bool>(outcome);
      }));
  if (numMeasuredTargets == 0) {
    return failure();
  }

  SmallVector<Value> quantumControls;
  SmallVector<size_t> quantumControlIndices;
  SmallVector<Value> measuredControlOutcomes;
  for (auto [index, control] : llvm::enumerate(op.getControlsIn())) {
    if (Value outcome = getPredecessorMeasurementOutcome(control)) {
      measuredControlOutcomes.push_back(outcome);
    } else {
      quantumControls.push_back(control);
      quantumControlIndices.push_back(index);
    }
  }

  if (numMeasuredTargets == 2 && quantumControls.empty()) {
    replaceRZZCtrlOp(op, *targetResultOrder, op.getControlsIn(),
                     op.getTargetsIn(), rewriter);
    return success();
  }

  utils::hoistSupportingOpsBefore(*op.getBody(), rzzOp, op, rewriter);
  const Value theta = rzzOp.getTheta();
  rewriter.setInsertionPoint(op);

  SmallVector<Value> transformedControls(op.getNumControls());
  SmallVector<Value> transformedTargets;
  if (measuredControlOutcomes.empty()) {
    auto transformed =
        buildMeasuredRZZReplacement(rewriter, op.getLoc(), quantumControls,
                                    op.getTargetsIn(), targetOutcomes, theta);
    transformedControls.assign(transformed.begin(),
                               transformed.begin() + op.getNumControls());
    transformedTargets.assign(transformed.begin() + op.getNumControls(),
                              transformed.end());
  } else {
    Value condition = measuredControlOutcomes.front();
    for (const Value outcome : llvm::drop_begin(measuredControlOutcomes)) {
      condition =
          arith::AndIOp::create(rewriter, op.getLoc(), condition, outcome);
    }
    SmallVector<Value> ifOperands(quantumControls);
    llvm::append_range(ifOperands, op.getTargetsIn());
    auto ifOp = IfOp::create(
        rewriter, op.getLoc(), condition, ifOperands,
        [&](ValueRange qubits) -> SmallVector<Value> {
          return buildMeasuredRZZReplacement(
              rewriter, op.getLoc(), qubits.take_front(quantumControls.size()),
              qubits.drop_front(quantumControls.size()), targetOutcomes, theta);
        });

    size_t quantumIndex = 0;
    for (const size_t controlIndex : llvm::seq<size_t>(op.getNumControls())) {
      if (quantumIndex < quantumControlIndices.size() &&
          quantumControlIndices[quantumIndex] == controlIndex) {
        transformedControls[controlIndex] =
            ifOp.getLinearResults()[quantumIndex++];
      } else {
        transformedControls[controlIndex] = op.getInputControl(controlIndex);
      }
    }
    transformedTargets.assign(
        ifOp.getLinearResults().drop_front(quantumControls.size()).begin(),
        ifOp.getLinearResults().drop_front(quantumControls.size()).end());
  }

  replaceRZZCtrlOp(op, *targetResultOrder, transformedControls,
                   transformedTargets, rewriter);
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
