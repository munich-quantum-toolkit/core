/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Rewriter.hpp"

#include "ConstantPropagationAnalysis.hpp"
#include "Decisions.hpp"
#include "UnionTable.hpp"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>

namespace mlir::qco {

SmallVector<Decision> collectDecisions(func::FuncOp entry,
                                       DataFlowSolver& solver) {
  SmallVector<Decision> decisions;

  entry.walk([&](CtrlOp op) {
    // A controlled gate nested in another modifier's body is interpreted by
    // that modifier's handler and not rewritten.
    if (op->getParentOfType<CtrlOp>() || op->getParentOfType<InvOp>() ||
        op->getParentOfType<PowOp>()) {
      return;
    }

    const auto* lattice =
        solver.lookupState<UnionTableLattice>(solver.getProgramPointBefore(op));
    if (lattice == nullptr || !lattice->isInitialized()) {
      return;
    }
    const UnionTable& table = lattice->getUnionTable();
    if (table.isAllTop()) {
      return;
    }

    const SmallVector<Value> controls(op.getInputControls().begin(),
                                      op.getInputControls().end());

    if (!table.areControlsSatisfiable(controls)) {
      decisions.push_back(DropOp{op});
      return;
    }

    const SuperfluousResult superfluous =
        table.getSuperfluousControls(controls);
    if (superfluous.completelySuperfluous) {
      decisions.push_back(DropOp{op});
      return;
    }

    SmallVector<unsigned> dropIndices;
    for (const auto& [index, control] : llvm::enumerate(controls)) {
      if (superfluous.superfluousQubits.contains(control)) {
        dropIndices.push_back(static_cast<unsigned>(index));
      }
    }
    // A strict subset is stripped; all of them means the gate fires
    // unconditionally and its body is inlined (see applyStrip).
    if (!dropIndices.empty()) {
      decisions.push_back(StripControls{op, std::move(dropIndices)});
    }
  });

  return decisions;
}

/// @brief Erases a never-firing controlled gate: every output qubit is replaced
/// by the matching input.
static void applyDrop(const DropOp& drop, IRRewriter& rewriter) {
  CtrlOp op = drop.op;
  for (auto [in, out] :
       llvm::zip_equal(op.getInputQubits(), op.getOutputQubits())) {
    rewriter.replaceAllUsesWith(out, in);
  }
  rewriter.eraseOp(op);
}

/// @brief Removes always-satisfied controls from a controlled gate.
///
/// A CtrlOp's body block arguments alias its *targets* only - controls merely
/// pass through - so dropping a subset just rebuilds the op around the same
/// body. Dropping every control means the body runs unconditionally: it is
/// inlined in place of the op, with the target block arguments bound to the
/// target operands and the yielded values taking over the op's target results.
static void applyStrip(const StripControls& strip, IRRewriter& rewriter) {
  CtrlOp op = strip.op;
  const auto controlsIn = op.getInputControls();
  const auto isDropped = [&](size_t index) {
    return llvm::is_contained(strip.dropControlIndices,
                              static_cast<unsigned>(index));
  };

  SmallVector<Value> keptControls;
  for (const auto& [index, control] : llvm::enumerate(controlsIn)) {
    if (!isDropped(index)) {
      keptControls.push_back(control);
    }
  }

  rewriter.setInsertionPoint(op);

  if (keptControls.empty()) {
    Block& body = op.getRegion().front();
    auto yield = cast<YieldOp>(body.getTerminator());
    const auto yielded = yield.getOperands();
    rewriter.inlineBlockBefore(&body, op, op.getInputTargets());
    for (auto [result, value] :
         llvm::zip_equal(op.getOutputTargets(), yielded)) {
      rewriter.replaceAllUsesWith(result, value);
    }
    for (auto [result, control] :
         llvm::zip_equal(op.getOutputControls(), controlsIn)) {
      rewriter.replaceAllUsesWith(result, control);
    }
    rewriter.eraseOp(yield);
    rewriter.eraseOp(op);
    return;
  }

  auto newOp =
      CtrlOp::create(rewriter, op.getLoc(), keptControls, op.getInputTargets());
  rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                              newOp.getRegion().end());

  for (const auto& [index, control] : llvm::enumerate(controlsIn)) {
    rewriter.replaceAllUsesWith(
        op.getOutputControl(index),
        isDropped(index) ? control : newOp.getOutputForInput(control));
  }
  for (const auto& [index, target] : llvm::enumerate(op.getInputTargets())) {
    rewriter.replaceAllUsesWith(op.getOutputTarget(index),
                                newOp.getOutputForInput(target));
  }
  rewriter.eraseOp(op);
}

void applyDecisions(ArrayRef<Decision> decisions, IRRewriter& rewriter) {
  for (const Decision& decision : decisions) {
    if (const auto* drop = std::get_if<DropOp>(&decision)) {
      applyDrop(*drop, rewriter);
    } else {
      applyStrip(std::get<StripControls>(decision), rewriter);
    }
  }
}

} // namespace mlir::qco
