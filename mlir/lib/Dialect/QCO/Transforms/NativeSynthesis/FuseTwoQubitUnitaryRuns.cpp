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
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace mlir::qco {

#define GEN_PASS_DEF_FUSETWOQUBITUNITARYRUNS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

static void
collectUnitaryOpsInPreOrder(Operation* root,
                            llvm::SmallVectorImpl<Operation*>& ops) {
  root->walk([&](Operation* op) {
    if (op->getParentOfType<CtrlOp>()) {
      return;
    }
    if (!llvm::isa<InvOp>(op) && op->getParentOfType<InvOp>()) {
      return;
    }
    if (llvm::isa<UnitaryOpInterface>(op)) {
      ops.push_back(op);
    }
  });
}

static Value emitSingleQubitMatrix(IRRewriter& rewriter, Location loc,
                                   Value inQubit, const Matrix2x2& matrix,
                                   const decomposition::EulerBasis basis) {
  // Force emission (`hasNonBasisGate = true`, `runSize = 0`) so the matrix is
  // always lowered into native gates of `basis`, including any residual
  // `qco.gphase`. With these arguments `synthesizeUnitary1QEuler` never
  // returns `std::nullopt`.
  return *decomposition::synthesizeUnitary1QEuler(
      rewriter, loc, inQubit, matrix, /*runSize=*/0,
      /*hasNonBasisGate=*/true, basis);
}

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
  void process(Operation* op, const decomposition::NativeProfileSpec& spec);
  LogicalResult materialize(IRRewriter& rewriter,
                            const decomposition::NativeProfileSpec& spec);
};

/// Map a single-qubit `UnitaryOpInterface` op to the
/// `decomposition::NativeGateKind` that must appear in the menu for the op to
/// be a no-op.
static std::optional<decomposition::NativeGateKind>
singleQubitNativeGateKind(UnitaryOpInterface op) {
  Operation* raw = op.getOperation();
  if (llvm::isa<UOp>(raw)) {
    return decomposition::NativeGateKind::U;
  }
  if (llvm::isa<XOp>(raw)) {
    return decomposition::NativeGateKind::X;
  }
  if (llvm::isa<SXOp>(raw)) {
    return decomposition::NativeGateKind::Sx;
  }
  if (llvm::isa<RZOp, POp>(raw)) {
    // `p` is a Z-rotation primitive for menu purposes.
    return decomposition::NativeGateKind::Rz;
  }
  if (llvm::isa<RXOp>(raw)) {
    return decomposition::NativeGateKind::Rx;
  }
  if (llvm::isa<RYOp>(raw)) {
    return decomposition::NativeGateKind::Ry;
  }
  if (llvm::isa<ROp>(raw)) {
    return decomposition::NativeGateKind::R;
  }
  return std::nullopt;
}

// NOLINTNEXTLINE(misc-use-internal-linkage): test-visible (see comment above).
static bool allowsSingleQubitOp(UnitaryOpInterface op,
                                const decomposition::NativeProfileSpec& spec) {
  if (llvm::isa<BarrierOp, GPhaseOp>(op.getOperation())) {
    return true;
  }
  const auto gate = singleQubitNativeGateKind(op);
  return gate && spec.allowedGates.contains(*gate);
}

static bool getBlockTwoQubitMatrix(Operation* op, Matrix4x4& matrix) {
  if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
    return false;
  }
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    auto* body = ctrl.getBodyUnitary(0).getOperation();
    if (llvm::isa<XOp>(body)) {
      matrix = Matrix4x4::fromElements(1, 0, 0, 0, //
                                       0, 1, 0, 0, //
                                       0, 0, 0, 1, //
                                       0, 0, 1, 0);
      return true;
    }
    if (llvm::isa<ZOp>(body)) {
      matrix = Matrix4x4::identity();
      matrix(3, 3) = -1.0;
      return true;
    }
    return false;
  }
  auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
  if (!unitary || !unitary.isTwoQubit()) {
    return false;
  }
  Matrix4x4 raw;
  if (!unitary.getUnitaryMatrix4x4(raw)) {
    return false;
  }
  matrix = raw;
  return true;
}

/// Check whether a two-qubit op `op` is already expressible by the resolved
/// native menu: a single-control `CX`/`CZ` consistent with the active
/// entangler, or `Rzz` when `spec.allowRzz` is set. Multi-control and other
/// two-qubit ops are considered non-native.
static bool isNativeTwoQubitOp(Operation* op,
                               const decomposition::NativeProfileSpec& spec) {
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    auto* body = ctrl.getBodyUnitary(0).getOperation();
    if (llvm::isa<XOp>(body)) {
      return llvm::is_contained(spec.entanglerBases,
                                decomposition::EntanglerBasis::Cx);
    }
    if (llvm::isa<ZOp>(body)) {
      return llvm::is_contained(spec.entanglerBases,
                                decomposition::EntanglerBasis::Cz);
    }
    return false;
  }
  return spec.allowRzz && llvm::isa<RZZOp>(op);
}

/// Decide whether replacing a consolidated window is worthwhile. Always
/// replace a window that contains any non-native op (we have to lower them
/// anyway); otherwise only replace when the deterministic synthesizer uses
/// strictly fewer entanglers than the window already contains.
static bool shouldApplyBlockReplacement(const TwoQubitBlock& block,
                                        std::uint8_t numBasisUses) {
  if (block.anyNonNative) {
    return true;
  }
  return numBasisUses < block.numTwoQ;
}

static LogicalResult
materializeSingleTwoQubitBlock(IRRewriter& rewriter, const TwoQubitBlock& block,
                               const decomposition::NativeProfileSpec& spec) {
  Operation* firstOp = block.ops.front();
  auto firstUnitary = llvm::cast<UnitaryOpInterface>(firstOp);
  const Value inA = firstUnitary.getInputQubit(0);
  const Value inB = firstUnitary.getInputQubit(1);
  const Value outA = block.wireA;
  const Value outB = block.wireB;

  rewriter.setInsertionPoint(firstOp);
  Value newA;
  Value newB;
  if (failed(decomposition::synthesizeUnitary2QWeyl(rewriter, firstOp->getLoc(),
                                                    inA, inB, block.accum, spec,
                                                    newA, newB))) {
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

void TwoQubitWindowConsolidator::process(
    Operation* op, const decomposition::NativeProfileSpec& spec) {
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
      llvm::SmallVector<std::size_t, 2> ids;
      if (v0 == block.wireA && v1 == block.wireB) {
        ids = {0, 1};
      } else if (v0 == block.wireB && v1 == block.wireA) {
        ids = {1, 0};
      } else {
        closeBlock(idx);
        return;
      }
      block.accum =
          reorderTwoQubitMatrix(opMatrix, ids[0], ids[1]) * block.accum;
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
    const auto pad = (v == block.wireA) ? embedSingleQubitInTwoQubit(raw, 0)
                                        : embedSingleQubitInTwoQubit(raw, 1);
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

LogicalResult TwoQubitWindowConsolidator::materialize(
    IRRewriter& rewriter, const decomposition::NativeProfileSpec& spec) {
  llvm::DenseSet<Operation*> erasedOps;
  for (const auto& block : blocks) {
    if (block.ops.size() < 2) {
      continue;
    }
    if (llvm::any_of(block.ops,
                     [&](Operation* op) { return erasedOps.contains(op); })) {
      continue;
    }
    const auto numBasisUses =
        decomposition::twoQubitEntanglerCount(block.accum, spec);
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

static LogicalResult
fuseTwoQubitUnitaryRuns(IRRewriter& rewriter, Operation* root,
                        const decomposition::NativeProfileSpec& spec) {
  llvm::SmallVector<Operation*, 32> ops;
  collectUnitaryOpsInPreOrder(root, ops);
  TwoQubitWindowConsolidator consolidator;
  for (Operation* op : ops) {
    consolidator.process(op, spec);
  }
  return consolidator.materialize(rewriter, spec);
}

/// Adjacent single-qubit unitaries on one wire considered for fusion.
struct OneQubitRun {
  llvm::SmallVector<UnitaryOpInterface, 4> ops;
};

/// If profitable, replace the run with one synthesized single-qubit op in
/// `basis` (mirrors `FuseSingleQubitUnitaryRuns`). Fuses when any op is
/// off-menu or when Euler resynthesis strictly shortens the run.
static bool maybeFuseRun(IRRewriter& rewriter, OneQubitRun& run,
                         const decomposition::EulerBasis basis,
                         const decomposition::NativeProfileSpec& spec) {
  Matrix2x2 fused = Matrix2x2::identity();
  for (UnitaryOpInterface u : run.ops) {
    Matrix2x2 m;
    if (!u.getUnitaryMatrix2x2(m)) {
      return false;
    }
    fused.premultiplyBy(m);
  }

  const bool anyNonNative = llvm::any_of(run.ops, [&](UnitaryOpInterface u) {
    return !allowsSingleQubitOp(u, spec);
  });

  Operation* firstOp = run.ops.front().getOperation();
  const Value inQubit = run.ops.front().getInputQubit(0);
  const Value outQubit = run.ops.back().getOutputQubit(0);

  rewriter.setInsertionPoint(firstOp);
  const auto replacement = decomposition::synthesizeUnitary1QEuler(
      rewriter, firstOp->getLoc(), inQubit, fused, run.ops.size(), anyNonNative,
      basis);
  if (!replacement) {
    return false;
  }
  rewriter.replaceAllUsesWith(outQubit, *replacement);
  for (auto& op : llvm::reverse(run.ops)) {
    rewriter.eraseOp(op.getOperation());
  }
  return true;
}

/// True when `op` lives in a `ctrl`/`inv` region body (not the shell op).
/// Skips nested unitaries so they are handled via the enclosing modifier.
static bool isHiddenInsideCtrlOrInvBody(Operation* op) {
  if (op->getParentOfType<CtrlOp>()) {
    return true;
  }
  if (!llvm::isa<InvOp>(op) && op->getParentOfType<InvOp>()) {
    return true;
  }
  return false;
}

/// Single-qubit op eligible for fusion (constant `2×2`, not under `ctrl`).
static UnitaryOpInterface fusibleSingleQubitOp(Operation* op) {
  auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
  if (!unitary || !unitary.isSingleQubit()) {
    return {};
  }
  if (llvm::isa<BarrierOp, GPhaseOp, CtrlOp>(op)) {
    return {};
  }
  if (isHiddenInsideCtrlOrInvBody(op)) {
    return {};
  }
  Matrix2x2 matrix;
  if (!unitary.getUnitaryMatrix2x2(matrix)) {
    return {};
  }
  return unitary;
}

/// Lowers unitary QCO ops to a comma-separated native gate menu using a
/// deterministic, matrix-driven synthesizer: single-qubit fuse, two-qubit
/// window consolidation, synthesis sweeps, seam single-qubit fuse, and
/// optional cleanup sweeps.
struct FuseTwoQubitUnitaryRunsPass
    : impl::FuseTwoQubitUnitaryRunsBase<FuseTwoQubitUnitaryRunsPass> {
  /// Default-construct the pass with the TableGen-generated option defaults.
  FuseTwoQubitUnitaryRunsPass() = default;

  explicit FuseTwoQubitUnitaryRunsPass(FuseTwoQubitUnitaryRunsOptions options)
      : FuseTwoQubitUnitaryRunsBase(std::move(options)) {}

protected:
  /// Top-level pass entry point. Resolves the native-gate menu, then drives
  /// the staged rewrite pipeline: one-qubit run fusion, two-qubit window
  /// consolidation, synthesis sweeps until the single-qubit surface is native,
  /// seam cleanup, and a final fusion pass. Fails the pass on invalid input or
  /// non-convergence.
  void runOnOperation() override {
    // Empty native-gates string: no-op.
    if (llvm::StringRef(nativeGates).trim().empty()) {
      return;
    }
    auto specOpt = decomposition::parseNativeSpec(nativeGates);
    if (!specOpt) {
      getOperation().emitError()
          << "unsupported native gate menu (native-gates='" << nativeGates
          << "')";
      signalPassFailure();
      return;
    }
    const auto& spec = *specOpt;
    // Deterministic single-qubit basis: the first emitter drives all matrix
    // synthesis and run fusion.
    const decomposition::EulerBasis oneQubitBasis =
        decomposition::emitterEulerBasis(spec.singleQubitEmitters.front());

    IRRewriter rewriter(&getContext());

    fuseOneQubitRuns(rewriter, spec, oneQubitBasis);
    if (failed(consolidateTwoQubitBlocks(rewriter, spec))) {
      signalPassFailure();
      return;
    }
    // Two-qubit lowering can emit off-menu single-qubit ops (e.g. `rx`/`ry`);
    // repeat until clean or hit the sweep cap before seam cleanup.
    constexpr unsigned kMaxSynthesisSweeps = 4;
    for (unsigned i = 0; i < kMaxSynthesisSweeps; ++i) {
      if (failed(synthesizeRemainingOps(rewriter, spec, oneQubitBasis))) {
        signalPassFailure();
        return;
      }
      if (!hasNonNativeSingleQubitOps(spec)) {
        break;
      }
    }
    if (hasNonNativeSingleQubitOps(spec)) {
      getOperation().emitError()
          << "native gate synthesis did not converge within "
          << kMaxSynthesisSweeps
          << " sweeps (single-qubit ops remain outside the native menu)";
      signalPassFailure();
      return;
    }
    // Fuse single-qubit seams between two-qubit blocks.
    fuseOneQubitRuns(rewriter, spec, oneQubitBasis);
    // Re-check full menu (single-qubit ops, native `ctrl`, allowed bare `rzz`).
    constexpr unsigned kPostMenuCleanupSweeps = 4;
    unsigned postMenuSweepsRemaining = kPostMenuCleanupSweeps;
    while (hasNonNativeMenuOps(spec) && postMenuSweepsRemaining-- > 0) {
      if (failed(synthesizeRemainingOps(rewriter, spec, oneQubitBasis))) {
        signalPassFailure();
        return;
      }
      fuseOneQubitRuns(rewriter, spec, oneQubitBasis);
    }
    if (hasNonNativeMenuOps(spec)) {
      getOperation().emitError()
          << "native gate synthesis: operations remain outside the native menu "
             "after final cleanup";
      signalPassFailure();
      return;
    }
  }

  /// `CtrlOp` is already on-menu when the body is `X`/`Z` and the profile
  /// supplies `cx` / `cz` entanglers.
  static bool
  ctrlMatchesNativeMenu(CtrlOp ctrl,
                        const decomposition::NativeProfileSpec& spec) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    Operation* body = ctrl.getBodyUnitary(0).getOperation();
    const bool hasCX = llvm::isa<XOp>(body);
    const bool hasCZ = llvm::isa<ZOp>(body);
    if (!hasCX && !hasCZ) {
      return false;
    }
    return (llvm::is_contained(spec.entanglerBases,
                               decomposition::EntanglerBasis::Cx) &&
            hasCX) ||
           (llvm::is_contained(spec.entanglerBases,
                               decomposition::EntanglerBasis::Cz) &&
            hasCZ);
  }

  /// Bare two-qubit on-menu: `rzz` when the profile allows it.
  static bool
  bareTwoQubitMatchesNativeMenu(Operation* op,
                                const decomposition::NativeProfileSpec& spec) {
    return llvm::isa<RZZOp>(op) && spec.allowRzz &&
           spec.allowedGates.contains(decomposition::NativeGateKind::Rzz);
  }

  /// True if any unitary is outside `spec` (single-qubit, `ctrl`, or bare
  /// `rzz`).
  bool hasNonNativeMenuOps(const decomposition::NativeProfileSpec& spec) {
    const mlir::WalkResult walkResult =
        getOperation()->walk([&](Operation* op) {
          if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
            return mlir::WalkResult::advance();
          }
          if (isHiddenInsideCtrlOrInvBody(op)) {
            return mlir::WalkResult::advance();
          }
          if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
            if (!ctrlMatchesNativeMenu(ctrl, spec)) {
              return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
          }
          auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
          if (!unitary) {
            return mlir::WalkResult::advance();
          }
          if (unitary.isSingleQubit()) {
            if (!allowsSingleQubitOp(unitary, spec)) {
              return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
          }
          if (unitary.isTwoQubit()) {
            if (!bareTwoQubitMatchesNativeMenu(op, spec)) {
              return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
          }
          return mlir::WalkResult::interrupt();
        });
    return walkResult.wasInterrupted();
  }

  /// Any off-menu single-qubit unitary (ignores `ctrl` region bodies).
  bool
  hasNonNativeSingleQubitOps(const decomposition::NativeProfileSpec& spec) {
    const mlir::WalkResult walkResult =
        getOperation()->walk([&](Operation* op) {
          if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
            return mlir::WalkResult::advance();
          }
          if (isHiddenInsideCtrlOrInvBody(op)) {
            return mlir::WalkResult::advance();
          }
          auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
          if (!unitary || !unitary.isSingleQubit()) {
            return mlir::WalkResult::advance();
          }
          if (!allowsSingleQubitOp(unitary, spec)) {
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });
    return walkResult.wasInterrupted();
  }

private:
  /// Fuse adjacent single-qubit runs when the emitter wins on length or any op
  /// is off-menu.
  void fuseOneQubitRuns(IRRewriter& rewriter,
                        const decomposition::NativeProfileSpec& spec,
                        const decomposition::EulerBasis basis) {
    llvm::SmallVector<OneQubitRun> runs;
    llvm::DenseMap<Operation*, size_t> tailOpToRun;

    // Extend the current run only when this op consumes the run's *tail*
    // output with no other uses: both the `tailOpToRun` lookup and
    // `inQubit.hasOneUse()` are required. Without the single-use check a run
    // could fuse gates on a wire that also feeds another path (fan-out),
    // which would silently drop the sibling user.
    getOperation()->walk([&](Operation* op) {
      auto unitary = fusibleSingleQubitOp(op);
      if (!unitary) {
        return;
      }
      Value inQubit = unitary.getInputQubit(0);
      Operation* defOp = inQubit.getDefiningOp();
      auto it =
          (defOp != nullptr) ? tailOpToRun.find(defOp) : tailOpToRun.end();
      const bool canExtend = it != tailOpToRun.end() && inQubit.hasOneUse();
      if (canExtend) {
        const size_t runIdx = it->second;
        runs[runIdx].ops.push_back(unitary);
        tailOpToRun.erase(it);
        tailOpToRun[op] = runIdx;
      } else {
        runs.push_back(OneQubitRun{});
        runs.back().ops.push_back(unitary);
        tailOpToRun[op] = runs.size() - 1;
      }
    });

    for (auto& run : runs) {
      if (run.ops.size() < 2) {
        continue;
      }
      (void)maybeFuseRun(rewriter, run, basis, spec);
    }
  }

  /// Two-qubit windows with absorbed single-qubit ops: replace when a cheaper
  /// native sequence exists.
  LogicalResult
  consolidateTwoQubitBlocks(IRRewriter& rewriter,
                            const decomposition::NativeProfileSpec& spec) {
    return fuseTwoQubitUnitaryRuns(rewriter, getOperation(), spec);
  }

  /// One synthesis sweep over the whole function: rewrite every remaining
  /// off-menu unitary by dispatching to `rewriteSingleQubit` /
  /// `rewriteControlled` / `rewriteTwoQubit`. Returns `failure()` as soon as
  /// any op cannot be lowered to the native menu. Safe to call repeatedly;
  /// `runOnOperation` iterates until convergence.
  LogicalResult
  synthesizeRemainingOps(IRRewriter& rewriter,
                         const decomposition::NativeProfileSpec& spec,
                         const decomposition::EulerBasis basis) {
    llvm::SmallVector<Operation*, 32> ops;
    collectUnitaryOpsInPreOrder(getOperation(), ops);
    llvm::DenseSet<Operation*> erasedOps;

    for (Operation* op : ops) {
      // Pointers were collected before this loop; avoid dereferencing ops
      // erased by earlier rewrites in this same sweep.
      if (erasedOps.contains(op)) {
        continue;
      }
      // Nested regions under `ctrl` / `inv` are handled on the shell op
      // (e.g. `ctrl { inv { ... } }`, `inv { ... }`).
      if (isHiddenInsideCtrlOrInvBody(op)) {
        continue;
      }
      if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
        continue;
      }
      auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
      if (!unitary) {
        continue;
      }

      if (unitary.isSingleQubit()) {
        if (!allowsSingleQubitOp(unitary, spec)) {
          if (failed(rewriteSingleQubit(rewriter, op, unitary, basis))) {
            return failure();
          }
          erasedOps.insert(op);
        }
        continue;
      }

      if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
        const bool wasAlreadyNative = ctrlMatchesNativeMenu(ctrl, spec);
        if (failed(rewriteControlled(rewriter, ctrl, spec))) {
          return failure();
        }
        if (!wasAlreadyNative) {
          erasedOps.insert(op);
        }
        continue;
      }

      if (unitary.isTwoQubit()) {
        if (failed(rewriteTwoQubit(rewriter, op, unitary, spec))) {
          return failure();
        }
        erasedOps.insert(op);
        continue;
      }
    }
    return success();
  }

  /// Lower one off-menu single-qubit `op` via its constant `2×2` matrix and
  /// the Euler synthesizer in `basis`.
  static LogicalResult
  rewriteSingleQubit(IRRewriter& rewriter, Operation* op,
                     UnitaryOpInterface unitary,
                     const decomposition::EulerBasis basis) {
    rewriter.setInsertionPoint(op);
    const Value in = unitary.getInputQubit(0);
    Matrix2x2 matrix;
    if (!unitary.getUnitaryMatrix2x2(matrix)) {
      op->emitError("single-qubit operation with non-constant parameters is "
                    "not supported for native synthesis");
      return failure();
    }
    const Value replaced =
        emitSingleQubitMatrix(rewriter, op->getLoc(), in, matrix, basis);
    rewriter.replaceOp(op, replaced);
    return success();
  }

  /// Lower a single-control, single-target `CtrlOp` to the native profile.
  /// Fast-path: already-native `CX`/`CZ` are kept as-is. Otherwise, lift the
  /// controlled op to its 4x4 matrix and run the deterministic two-qubit
  /// synthesizer.
  static LogicalResult
  rewriteControlled(IRRewriter& rewriter, CtrlOp ctrl,
                    const decomposition::NativeProfileSpec& spec) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      ctrl.emitError("native synthesis currently only supports 1-control "
                     "1-target controlled gates");
      return failure();
    }
    auto* body = ctrl.getBodyUnitary(0).getOperation();
    const bool hasCX = llvm::isa<XOp>(body);
    const bool hasCZ = llvm::isa<ZOp>(body);
    if ((llvm::is_contained(spec.entanglerBases,
                            decomposition::EntanglerBasis::Cx) &&
         hasCX) ||
        (llvm::is_contained(spec.entanglerBases,
                            decomposition::EntanglerBasis::Cz) &&
         hasCZ)) {
      return success();
    }
    Matrix4x4 matrix;
    if (hasCX || hasCZ) {
      if (!getBlockTwoQubitMatrix(ctrl.getOperation(), matrix)) {
        ctrl.emitError("failed to compute 4x4 matrix for CtrlOp");
        return failure();
      }
    } else {
      auto u = llvm::cast<UnitaryOpInterface>(ctrl.getOperation());
      if (!u.isTwoQubit() || !u.getUnitaryMatrix4x4(matrix)) {
        ctrl.emitError(
            "native synthesis: cannot build a constant 4x4 matrix for this "
            "controlled gate (unsupported body or non-constant parameters)");
        return failure();
      }
    }
    rewriter.setInsertionPoint(ctrl);
    Value out0;
    Value out1;
    if (failed(decomposition::synthesizeUnitary2QWeyl(
            rewriter, ctrl.getLoc(), ctrl.getInputControl(0),
            ctrl.getInputTarget(0), matrix, spec, out0, out1))) {
      ctrl.emitError("controlled gate not allowed by selected profile");
      return failure();
    }
    rewriter.replaceOp(ctrl, ValueRange{out0, out1});
    return success();
  }

  /// Lower an off-menu generic two-qubit op. Bare `RZZ` is kept when on the
  /// native menu; all other two-qubit unitaries go through the deterministic
  /// KAK synthesizer.
  static LogicalResult
  rewriteTwoQubit(IRRewriter& rewriter, Operation* op,
                  UnitaryOpInterface unitary,
                  const decomposition::NativeProfileSpec& spec) {
    if (spec.allowRzz && llvm::isa<RZZOp>(op)) {
      return success();
    }
    Matrix4x4 matrix;
    if (!getBlockTwoQubitMatrix(op, matrix)) {
      op->emitError("unsupported two-qubit operation for selected profile");
      return failure();
    }
    rewriter.setInsertionPoint(op);
    Value out0;
    Value out1;
    if (failed(decomposition::synthesizeUnitary2QWeyl(
            rewriter, op->getLoc(), unitary.getInputQubit(0),
            unitary.getInputQubit(1), matrix, spec, out0, out1))) {
      op->emitError("unsupported two-qubit operation for selected profile");
      return failure();
    }
    rewriter.replaceOp(op, ValueRange{out0, out1});
    return success();
  }
};

} // namespace mlir::qco
