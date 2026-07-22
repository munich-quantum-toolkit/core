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
    auto op = dyn_cast<CtrlOp>(*measure.getQubitOut().getUsers().begin());
    if (!op) {
      return failure();
    }
    rewriter.setInsertionPointAfter(op);

    if (utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody())) {
      trySwapControlsOfDiagonalGate(op, rewriter);
    }

    SmallVector<std::pair<Value, Value>> toReplace;
    SmallVector<Value> toKeep;

    for (const auto& operand : op.getControlsIn()) {
      auto outcome = getPredecessorMeasurementOutcome(operand);
      if (outcome) {
        toReplace.emplace_back(operand, outcome);
      } else {
        toKeep.push_back(operand);
      }
    }

    if (toReplace.empty()) {
      return failure();
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
