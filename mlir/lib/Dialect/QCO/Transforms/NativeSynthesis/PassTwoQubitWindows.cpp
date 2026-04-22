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
#include "mlir/Dialect/QCO/Transforms/Decomposition/UnitaryMatrices.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Policy.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Scoring.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/TwoQubit.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Utils.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Casting.h>

#include <ranges>

namespace mlir::qco::native_synth {
namespace {

bool isNativeTwoQubitOp(Operation* op, const NativeProfileSpec& spec) {
  if (auto ctrl = dyn_cast<CtrlOp>(op)) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    auto* body = ctrl.getBodyUnitary().getOperation();
    if (isa<XOp>(body)) {
      return usesCxEntangler(spec);
    }
    if (isa<ZOp>(body)) {
      return usesCzEntangler(spec);
    }
    return false;
  }
  return spec.allowRzz && isa<RZZOp>(op);
}

bool shouldApplyBlockReplacement(const TwoQubitBlock& block,
                                 const CandidateMetrics& best) {
  if (block.anyNonNative) {
    return true;
  }
  const bool shorterTwoQ = best.numTwoQ < block.numTwoQ;
  const bool sameTwoQ = best.numTwoQ == block.numTwoQ;
  const bool shorterOneQ = best.numOneQ < block.numOneQ;
  return shorterTwoQ || (sameTwoQ && shorterOneQ);
}

} // namespace

static void materializeSingleTwoQubitBlock(
    IRRewriter& rewriter, const TwoQubitBlock& block,
    const SynthesisCandidate<TwoQubitRewritePlan>& best) {
  Operation* firstOp = block.ops.front();
  auto firstUnitary = cast<UnitaryOpInterface>(firstOp);
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
    return;
  }
  if (best.payload.sequence.hasGlobalPhase()) {
    emitGPhaseIfNonTrivial(rewriter, firstOp->getLoc(),
                           best.payload.sequence.globalPhase);
  }
  rewriter.replaceAllUsesWith(outA, newA);
  rewriter.replaceAllUsesWith(outB, newB);
  for (auto* toErase : std::ranges::reverse_view(block.ops)) {
    rewriter.eraseOp(toErase);
  }
}

void collectUnitaryOpsInPreOrder(Operation* root,
                                 std::vector<Operation*>& ops) {
  root->walk([&](Operation* op) {
    if (isa<UnitaryOpInterface>(op)) {
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
  wireToBlock.erase(block.wireB);
}

void TwoQubitWindowConsolidator::closeBlockOnWire(Value v) {
  if (auto it = wireToBlock.find(v); it != wireToBlock.end()) {
    closeBlock(it->second);
  }
}

void TwoQubitWindowConsolidator::process(Operation* op,
                                         const NativeProfileSpec& spec) {
  if (isa_and_present<CtrlOp>(op->getParentOp())) {
    return;
  }
  auto unitary = dyn_cast<UnitaryOpInterface>(op);
  if (!unitary) {
    return;
  }
  if (isa<BarrierOp, GPhaseOp>(op)) {
    for (Value v : op->getOperands()) {
      closeBlockOnWire(v);
    }
    return;
  }

  if (unitary.isTwoQubit()) {
    Eigen::Matrix4cd opMatrix;
    if (!getBlockTwoQubitMatrix(op, opMatrix)) {
      closeBlockOnWire(unitary.getInputQubit(0));
      closeBlockOnWire(unitary.getInputQubit(1));
      return;
    }
    const Value v0 = unitary.getInputQubit(0);
    const Value v1 = unitary.getInputQubit(1);
    auto it0 = wireToBlock.find(v0);
    auto it1 = wireToBlock.find(v1);
    const bool tracked0 = it0 != wireToBlock.end();
    const bool tracked1 = it1 != wireToBlock.end();
    const bool sameBlock = tracked0 && tracked1 && it0->second == it1->second;
    const bool singleUse = v0.hasOneUse() && v1.hasOneUse();

    if (sameBlock && singleUse) {
      const size_t idx = it0->second;
      auto& block = blocks[idx];
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
      wireToBlock.erase(it1);
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

    if (tracked0) {
      closeBlock(it0->second);
    }
    if (tracked1 && (!tracked0 || it0->second != it1->second)) {
      closeBlock(it1->second);
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

  if (unitary.isSingleQubit()) {
    const Value v = unitary.getInputQubit(0);
    auto it = wireToBlock.find(v);
    if (it == wireToBlock.end()) {
      return;
    }
    const size_t idx = it->second;
    auto& block = blocks[idx];
    Eigen::Matrix2cd m;
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

  for (Value v : op->getOperands()) {
    closeBlockOnWire(v);
  }
}

void TwoQubitWindowConsolidator::materialize(IRRewriter& rewriter,
                                             const NativeProfileSpec& spec,
                                             const ScoreWeights& weights) {
  for (const auto& block : blocks) {
    if (block.ops.size() < 2) {
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
    materializeSingleTwoQubitBlock(rewriter, block, *best);
  }
}

} // namespace mlir::qco::native_synth
