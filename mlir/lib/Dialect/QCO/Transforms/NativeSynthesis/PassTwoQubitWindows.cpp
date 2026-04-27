/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/PassTwoQubitWindows.h"

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Gate.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/UnitaryMatrices.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Policy.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Scoring.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/TwoQubit.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Utils.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <optional>
#include <utility>

namespace mlir::qco::native_synth {

/// Check whether a two-qubit op `op` is already expressible by the resolved
/// native menu: a single-control `CX`/`CZ` consistent with the active
/// entangler, or `Rzz` when `spec.allowRzz` is set. Multi-control and other
/// two-qubit ops are considered non-native.
static bool isNativeTwoQubitOp(Operation* op, const NativeProfileSpec& spec) {
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    auto* body = ctrl.getBodyUnitary().getOperation();
    if (llvm::isa<XOp>(body)) {
      return usesCxEntangler(spec);
    }
    if (llvm::isa<ZOp>(body)) {
      return usesCzEntangler(spec);
    }
    return false;
  }
  return spec.allowRzz && llvm::isa<RZZOp>(op);
}

/// Decide whether replacing a consolidated window with the candidate
/// described by `best` is worthwhile. Always replace a window that contains
/// any non-native op (we have to lower them anyway); otherwise only replace
/// when the candidate has strictly fewer two-qubit gates, or the same number
/// with strictly fewer one-qubit gates.
static bool shouldApplyBlockReplacement(const TwoQubitBlock& block,
                                        const CandidateMetrics& best) {
  if (block.anyNonNative) {
    return true;
  }
  const bool shorterTwoQ = best.numTwoQ < block.numTwoQ;
  const bool sameTwoQ = best.numTwoQ == block.numTwoQ;
  const bool shorterOneQ = best.numOneQ < block.numOneQ;
  return shorterTwoQ || (sameTwoQ && shorterOneQ);
}

/// Emit the chosen synthesis sequence `best` at the location of the window's
/// first op, rewire the block's trailing SSA values (`wireA`, `wireB`) to
/// the newly emitted outputs, and erase the replaced ops in reverse order
/// so def-use edges are cleared before their defining ops disappear.
static LogicalResult materializeSingleTwoQubitBlock(
    IRRewriter& rewriter, const TwoQubitBlock& block,
    const SynthesisCandidate<TwoQubitRewritePlan>& best) {
  Operation* firstOp = block.ops.front();
  auto firstUnitary = llvm::cast<UnitaryOpInterface>(firstOp);
  const Value inA = firstUnitary.getInputQubit(0);
  const Value inB = firstUnitary.getInputQubit(1);
  const Value outA = block.wireA;
  const Value outB = block.wireB;

  rewriter.setInsertionPoint(firstOp);
  Value newA;
  Value newB;
  if (failed(emitTwoQubitGateSequenceAtLoc(rewriter, firstOp->getLoc(), inA,
                                           inB, best.payload.sequence, newA,
                                           newB))) {
    firstOp->emitError("failed to emit synthesized two-qubit gate sequence");
    return failure();
  }
  if (best.payload.sequence.hasGlobalPhase()) {
    emitGPhaseIfNonTrivial(rewriter, firstOp->getLoc(),
                           best.payload.sequence.globalPhase);
  }
  rewriter.replaceAllUsesWith(outA, newA);
  rewriter.replaceAllUsesWith(outB, newB);
  for (auto* toErase : llvm::reverse(block.ops)) {
    rewriter.eraseOp(toErase);
  }
  return success();
}

void collectUnitaryOpsInPreOrder(Operation* root,
                                 llvm::SmallVectorImpl<Operation*>& ops) {
  root->walk([&](Operation* op) {
    if (op->getParentOfType<CtrlOp>()) {
      return;
    }
    if (llvm::isa<UnitaryOpInterface>(op)) {
      ops.push_back(op);
    }
  });
}

void TwoQubitWindowConsolidator::closeBlock(size_t idx) {
  auto& block = blocks[idx];
  if (!block.open) {
    return;
  }
  block.open = false;
  wireToBlock.erase(block.wireA);
  if (block.wireB != block.wireA) {
    wireToBlock.erase(block.wireB);
  }
}

void TwoQubitWindowConsolidator::closeBlockOnWire(Value v) {
  if (auto it = wireToBlock.find(v); it != wireToBlock.end()) {
    closeBlock(it->second);
  }
}

/// State-machine step for one IR op, invoked in walk order over the module.
///
/// The consolidator tracks a set of *maximal two-qubit windows* -- contiguous
/// slices of the dataflow where at most two qubit wires interact -- so a
/// later pass can re-synthesize each window as a single 4x4 unitary. For
/// each op we update two pieces of state:
///
///   * `blocks`        -- append-only list of `TwoQubitBlock`s. Closed
///                        blocks are kept so `materialize()` can rewrite
///                        them later.
///   * `wireToBlock`   -- maps each *currently-open* SSA qubit Value to the
///                        index of the block that still owns it.
///                        Re-keyed whenever an op produces a new output
///                        Value on a tracked wire.
///
/// Because `process` is called in pre-order over the IR, when we see an op
/// its input Values have already been processed (or were function
/// arguments). A block stays open for a wire as long as every op consuming
/// that wire is either (a) a single-qubit op absorbable into the block, or
/// (b) another two-qubit op on the *same* pair of wires. Any other
/// consumer -- a barrier, a control, a different pair of wires, a
/// multi-use fork -- closes the block.
void TwoQubitWindowConsolidator::process(Operation* op,
                                         const NativeProfileSpec& spec) {
  // Skip ops nested anywhere under a `CtrlOp` (e.g. `ctrl { inv { ... } }`):
  // those are handled as part of the enclosing controlled op, not as
  // independent gates for window tracking.
  if (op->getParentOfType<CtrlOp>()) {
    return;
  }
  auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
  if (!unitary) {
    return;
  }
  // Barriers and stand-alone global-phase ops are not unitaries we can
  // absorb; they act as synchronization points that force any block
  // touching their operand wires to close.
  if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
    for (Value v : op->getOperands()) {
      closeBlockOnWire(v);
    }
    return;
  }

  if (unitary.isTwoQubit()) {
    // A two-qubit op for which we cannot build a 4x4 matrix is opaque to the
    // window model; close any blocks on its inputs and bail out.
    Eigen::Matrix4cd opMatrix;
    if (!getBlockTwoQubitMatrix(op, opMatrix)) {
      closeBlockOnWire(unitary.getInputQubit(0));
      closeBlockOnWire(unitary.getInputQubit(1));
      return;
    }
    const Value v0 = unitary.getInputQubit(0);
    const Value v1 = unitary.getInputQubit(1);
    // Defensive guard: malformed/degenerated two-qubit ops with identical
    // input wires cannot be represented by this window model. Treat them as
    // synchronization points and avoid map-iterator aliasing UB below.
    if (v0 == v1) {
      closeBlockOnWire(v0);
      return;
    }
    auto it0 = wireToBlock.find(v0);
    auto it1 = wireToBlock.find(v1);
    const bool tracked0 = it0 != wireToBlock.end();
    const bool tracked1 = it1 != wireToBlock.end();
    const std::optional<size_t> idx0 =
        tracked0 ? std::optional(it0->second) : std::nullopt;
    const std::optional<size_t> idx1 =
        tracked1 ? std::optional(it1->second) : std::nullopt;
    // "Same block" means the two input wires are currently the (wireA,
    // wireB) pair of one existing block -- i.e. this op operates on the
    // same pair as the previous two-qubit op in that block. Otherwise the
    // op either extends into a *new* pair (merging two blocks, which we
    // don't support) or starts a fresh block.
    const bool sameBlock =
        idx0.has_value() && idx1.has_value() && *idx0 == *idx1;
    const bool singleUse = v0.hasOneUse() && v1.hasOneUse();

    // ---- Case A: extend the existing block ---------------------------
    // Both inputs belong to the same open block and nothing else uses
    // them. Absorb the new gate into the block's accumulated unitary and
    // advance the tracked wires to this op's outputs.
    if (sameBlock && singleUse) {
      const size_t idx = *idx0;
      auto& block = blocks[idx];
      // `block.accum` is the composite 4x4 unitary of the gates absorbed so
      // far, with qubit 0 == `wireA` and qubit 1 == `wireB`. The incoming
      // op's `opMatrix` is in the (v0, v1) operand order, so we reorder it
      // to the block's (wireA, wireB) convention before left-multiplying
      // (newest gate on the left, matching matrix-times-column-state order).
      llvm::SmallVector<decomposition::QubitId, 2> ids;
      if (v0 == block.wireA && v1 == block.wireB) {
        ids = {0, 1};
      } else if (v0 == block.wireB && v1 == block.wireA) {
        ids = {1, 0};
      } else {
        closeBlock(idx);
        return;
      }
      block.accum = decomposition::fixTwoQubitMatrixQubitOrder(opMatrix, ids) *
                    block.accum;
      block.ops.push_back(op);
      ++block.numTwoQ;
      if (!isNativeTwoQubitOp(op, spec)) {
        block.anyNonNative = true;
      }
      wireToBlock.erase(it0);
      if (it1 != it0) {
        wireToBlock.erase(it1);
      }
      Value newA;
      Value newB;
      if (v0 == block.wireA) {
        newA = unitary.getOutputQubit(0);
        newB = unitary.getOutputQubit(1);
      } else {
        newA = unitary.getOutputQubit(1);
        newB = unitary.getOutputQubit(0);
      }
      block.wireA = newA;
      block.wireB = newB;
      wireToBlock[newA] = idx;
      wireToBlock[newB] = idx;
      return;
    }

    // ---- Case B: close overlapping blocks, start a new one ----------
    // The inputs do not form a clean pair on an existing block (fan-out,
    // straddling two different blocks, or only one wire tracked). Closing
    // the affected blocks prevents wire-to-block aliasing from becoming
    // inconsistent -- note the second `if` guards against double-closing
    // the same block when both inputs happened to live in it but `sameBlock
    // && singleUse` was false (e.g. only fan-out violated).
    if (idx0.has_value()) {
      closeBlock(*idx0);
    }
    if (idx1.has_value() && (!idx0.has_value() || *idx0 != *idx1)) {
      closeBlock(*idx1);
    }
    TwoQubitBlock nb;
    nb.wireA = unitary.getOutputQubit(0);
    nb.wireB = unitary.getOutputQubit(1);
    nb.ops.push_back(op);
    nb.numTwoQ = 1;
    nb.accum = opMatrix;
    nb.anyNonNative = !isNativeTwoQubitOp(op, spec);
    const size_t idx = blocks.size();
    blocks.push_back(std::move(nb));
    wireToBlock[blocks[idx].wireA] = idx;
    wireToBlock[blocks[idx].wireB] = idx;
    return;
  }

  // ---- Case C: single-qubit op on a tracked wire -------------------
  // Absorbable into the block's accumulated 4x4 by lifting the 2x2 to the
  // appropriate tensor slot. If the wire is not tracked, the op simply
  // does not interact with any open block and is left for other passes.
  if (unitary.isSingleQubit()) {
    const Value v = unitary.getInputQubit(0);
    auto it = wireToBlock.find(v);
    if (it == wireToBlock.end()) {
      return;
    }
    const size_t idx = it->second;
    auto& block = blocks[idx];
    Eigen::Matrix2cd m;
    // `!v.hasOneUse()` is the fan-out guard: if any other op also consumes
    // this wire, we cannot soundly absorb this single-qubit gate into the
    // block (the sibling user would see the pre-gate state). Close the
    // block and let the outer pass rewrite the op individually.
    if (!unitary.getUnitaryMatrix2x2(m) || !v.hasOneUse()) {
      closeBlock(idx);
      return;
    }
    const auto pad = (v == block.wireA)
                         ? decomposition::expandToTwoQubits(m, 0)
                         : decomposition::expandToTwoQubits(m, 1);
    block.accum = pad * block.accum;
    block.ops.push_back(op);
    ++block.numOneQ;
    if (!allowsSingleQubitOp(unitary, spec)) {
      block.anyNonNative = true;
    }
    wireToBlock.erase(it);
    if (v == block.wireA) {
      block.wireA = unitary.getOutputQubit(0);
      wireToBlock[block.wireA] = idx;
    } else {
      block.wireB = unitary.getOutputQubit(0);
      wireToBlock[block.wireB] = idx;
    }
    return;
  }

  // ---- Case D: any other unitary (e.g. >2-qubit ops) ---------------
  // We can neither absorb nor continue a window through an op of unknown
  // arity, so close every block that touches one of its operand wires.
  for (Value v : op->getOperands()) {
    closeBlockOnWire(v);
  }
}

LogicalResult
TwoQubitWindowConsolidator::materialize(IRRewriter& rewriter,
                                        const NativeProfileSpec& spec,
                                        const ScoreWeights& weights) {
  llvm::DenseSet<Operation*> erasedOps;
  for (const auto& block : blocks) {
    if (block.ops.size() < 2) {
      continue;
    }
    // Rewriting earlier windows can erase ops captured in later windows.
    // Track erased op pointers and skip such windows without dereferencing
    // potentially dangling `Operation*`.
    if (llvm::any_of(block.ops,
                     [&](Operation* op) { return erasedOps.contains(op); })) {
      continue;
    }
    // Leave `block.accum` unnormalized: Weyl keeps stripped SU(4) phase in
    // the candidate sequence's `globalPhase`.
    const auto candidates =
        collectTwoQubitBasisCandidatesFromMatrix(block.accum, spec);
    const auto* best = selectBestCandidate(llvm::ArrayRef(candidates), weights);
    if (best == nullptr) {
      continue;
    }
    if (!shouldApplyBlockReplacement(block, best->metrics)) {
      continue;
    }
    if (failed(materializeSingleTwoQubitBlock(rewriter, block, *best))) {
      return failure();
    }
    for (Operation* op : block.ops) {
      erasedOps.insert(op);
    }
  }
  return success();
}

} // namespace mlir::qco::native_synth
