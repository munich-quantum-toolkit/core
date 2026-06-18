/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/FuseTwoQubitUnitaryRuns.h"

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/UnitaryMatrices.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/NativeSpec.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Policy.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/TwoQubit.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Utils.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace mlir::qco {

#define GEN_PASS_DEF_FUSETWOQUBITUNITARYRUNS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

} // namespace mlir::qco

namespace mlir::qco::native_synth {

namespace {

/// State for one maximal two-qubit window (plus absorbed one-qubit ops)
/// during consolidation.
struct TwoQubitBlock {
  Value wireA;
  Value wireB;
  llvm::SmallVector<Operation*, 8> ops;
  Matrix4x4 accum = Matrix4x4::identity();
  unsigned numTwoQ = 0;
  unsigned numOneQ = 0;
  bool anyNonNative = false;
  bool open = true;
};

/// Tracks overlapping two-qubit windows on a module slice.
struct TwoQubitWindowConsolidator {
  std::vector<TwoQubitBlock> blocks;
  llvm::DenseMap<Value, size_t> wireToBlock;

  void closeBlock(size_t idx);
  void closeBlockOnWire(Value v);
  void process(Operation* op, const NativeProfileSpec& spec);
  LogicalResult materialize(IRRewriter& rewriter,
                            const NativeProfileSpec& spec);
};

/// Check whether a two-qubit op `op` is already expressible by the resolved
/// native menu: a single-control `CX`/`CZ` consistent with the active
/// entangler, or `Rzz` when `spec.allowRzz` is set. Multi-control and other
/// two-qubit ops are considered non-native.
bool isNativeTwoQubitOp(Operation* op, const NativeProfileSpec& spec) {
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    auto* body = ctrl.getBodyUnitary(0).getOperation();
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

/// Decide whether replacing a consolidated window is worthwhile. Always
/// replace a window that contains any non-native op (we have to lower them
/// anyway); otherwise only replace when the deterministic synthesizer uses
/// strictly fewer entanglers than the window already contains.
bool shouldApplyBlockReplacement(const TwoQubitBlock& block,
                                 std::uint8_t numBasisUses) {
  if (block.anyNonNative) {
    return true;
  }
  return numBasisUses < block.numTwoQ;
}

LogicalResult materializeSingleTwoQubitBlock(IRRewriter& rewriter,
                                             const TwoQubitBlock& block,
                                             const NativeProfileSpec& spec) {
  Operation* firstOp = block.ops.front();
  auto firstUnitary = llvm::cast<UnitaryOpInterface>(firstOp);
  const Value inA = firstUnitary.getInputQubit(0);
  const Value inB = firstUnitary.getInputQubit(1);
  const Value outA = block.wireA;
  const Value outB = block.wireB;

  rewriter.setInsertionPoint(firstOp);
  Value newA;
  Value newB;
  if (failed(emitTwoQubitNative(rewriter, firstOp->getLoc(), inA, inB,
                                block.accum, spec, newA, newB))) {
    firstOp->emitError("failed to emit synthesized two-qubit gate sequence");
    return failure();
  }
  rewriter.replaceAllUsesWith(outA, newA);
  rewriter.replaceAllUsesWith(outB, newB);
  for (auto* toErase : llvm::reverse(block.ops)) {
    rewriter.eraseOp(toErase);
  }
  return success();
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

void TwoQubitWindowConsolidator::process(Operation* op,
                                         const NativeProfileSpec& spec) {
  if (op->getParentOfType<CtrlOp>()) {
    return;
  }
  if (!llvm::isa<InvOp>(op) && op->getParentOfType<InvOp>()) {
    return;
  }
  auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
  if (!unitary) {
    return;
  }
  if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
    for (Value v : op->getOperands()) {
      closeBlockOnWire(v);
    }
    return;
  }

  if (unitary.isTwoQubit()) {
    Matrix4x4 opMatrix;
    if (!getBlockTwoQubitMatrix(op, opMatrix)) {
      closeBlockOnWire(unitary.getInputQubit(0));
      closeBlockOnWire(unitary.getInputQubit(1));
      return;
    }
    const Value v0 = unitary.getInputQubit(0);
    const Value v1 = unitary.getInputQubit(1);
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
    const bool sameBlock =
        idx0.has_value() && idx1.has_value() && *idx0 == *idx1;
    const bool singleUse = v0.hasOneUse() && v1.hasOneUse();

    if (sameBlock && singleUse) {
      const size_t idx = *idx0;
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
      const Value eraseKeyA = it0->first;
      const Value eraseKeyB = it1->first;
      wireToBlock.erase(eraseKeyA);
      if (eraseKeyA != eraseKeyB) {
        wireToBlock.erase(eraseKeyB);
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

  if (unitary.isSingleQubit()) {
    const Value v = unitary.getInputQubit(0);
    auto it = wireToBlock.find(v);
    if (it == wireToBlock.end()) {
      return;
    }
    const size_t idx = it->second;
    auto& block = blocks[idx];
    Matrix2x2 raw;
    if (!unitary.getUnitaryMatrix2x2(raw) || !v.hasOneUse()) {
      closeBlock(idx);
      return;
    }
    const auto pad = (v == block.wireA)
                         ? decomposition::expandToTwoQubits(raw, 0)
                         : decomposition::expandToTwoQubits(raw, 1);
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

LogicalResult
TwoQubitWindowConsolidator::materialize(IRRewriter& rewriter,
                                        const NativeProfileSpec& spec) {
  llvm::DenseSet<Operation*> erasedOps;
  for (const auto& block : blocks) {
    if (block.ops.size() < 2) {
      continue;
    }
    if (llvm::any_of(block.ops,
                     [&](Operation* op) { return erasedOps.contains(op); })) {
      continue;
    }
    const auto numBasisUses = twoQubitEntanglerCount(block.accum, spec);
    if (!numBasisUses) {
      continue;
    }
    if (!shouldApplyBlockReplacement(block, *numBasisUses)) {
      continue;
    }
    if (failed(materializeSingleTwoQubitBlock(rewriter, block, spec))) {
      return failure();
    }
    for (Operation* op : block.ops) {
      erasedOps.insert(op);
    }
  }
  return success();
}

} // namespace

LogicalResult fuseTwoQubitUnitaryRuns(IRRewriter& rewriter, Operation* root,
                                      const NativeProfileSpec& spec) {
  llvm::SmallVector<Operation*, 32> ops;
  collectUnitaryOpsInPreOrder(root, ops);
  TwoQubitWindowConsolidator consolidator;
  for (Operation* op : ops) {
    consolidator.process(op, spec);
  }
  return consolidator.materialize(rewriter, spec);
}

namespace {

struct FuseTwoQubitUnitaryRunsPass final
    : impl::FuseTwoQubitUnitaryRunsBase<FuseTwoQubitUnitaryRunsPass> {
  using Base::Base;

  explicit FuseTwoQubitUnitaryRunsPass(FuseTwoQubitUnitaryRunsOptions options)
      : Base(std::move(options)) {}

protected:
  void runOnOperation() override {
    if (llvm::StringRef(nativeGates).trim().empty()) {
      return;
    }
    auto specOpt = resolveNativeGatesSpec(nativeGates);
    if (!specOpt) {
      getOperation().emitError()
          << "unsupported native gate menu (native-gates='" << nativeGates
          << "')";
      signalPassFailure();
      return;
    }
    IRRewriter rewriter(&getContext());
    if (failed(fuseTwoQubitUnitaryRuns(rewriter, getOperation(), *specOpt))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco::native_synth
