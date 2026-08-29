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
#include <mlir/IR/Value.h>

namespace mlir::qco {

SmallVector<Decision> collectDecisions(func::FuncOp entry,
                                       DataFlowSolver& solver) {
  SmallVector<Decision> decisions;

  entry.walk([&](CtrlOp op) {
    // A controlled gate nested in another modifier's body is interpreted by
    // that modifier's handler and not rewritten.
    if (isa<CtrlOp, InvOp, PowOp>(op->getParentOp())) {
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
    // Stripping *every* control would turn this into an uncontrolled gate -
    // deferred past v2.0. Only rebuild when a real control remains.
    if (!dropIndices.empty() && dropIndices.size() < controls.size()) {
      decisions.push_back(StripControls{op, std::move(dropIndices)});
    }
  });

  return decisions;
}

namespace {

/// @brief Erases a never-firing controlled gate: every output qubit is replaced
/// by the matching input.
void applyDrop(const DropOp& drop, IRRewriter& rewriter) {
  CtrlOp op = drop.op;
  for (auto [in, out] :
       llvm::zip_equal(op.getInputQubits(), op.getOutputQubits())) {
    rewriter.replaceAllUsesWith(out, in);
  }
  rewriter.eraseOp(op);
}

/// @brief Rebuilds a controlled gate with a subset of its controls. The body
/// region's block arguments alias the *targets* only, so it moves across
/// untouched.
void applyStrip(const StripControls& strip, IRRewriter& rewriter) {
  CtrlOp op = strip.op;
  const auto controlsIn = op.getInputControls();
  const auto isDropped = [&](const size_t index) {
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

} // namespace

void applyDecisions(const ArrayRef<Decision> decisions, IRRewriter& rewriter) {
  for (const Decision& decision : decisions) {
    if (const auto* drop = std::get_if<DropOp>(&decision)) {
      applyDrop(*drop, rewriter);
    } else {
      applyStrip(std::get<StripControls>(decision), rewriter);
    }
  }
}

} // namespace mlir::qco
