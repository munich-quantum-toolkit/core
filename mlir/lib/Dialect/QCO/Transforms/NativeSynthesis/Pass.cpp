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
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateSequence.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/NativeSpec.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/PassTwoQubitWindows.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Policy.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Scoring.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/SingleQubit.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/TwoQubit.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Utils.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <Eigen/Core>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>

#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <utility>

namespace mlir::qco {
#define GEN_PASS_DEF_NATIVEGATESYNTHESISPASS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"
} // namespace mlir::qco

namespace mlir::qco {

using native_synth::allowsSingleQubitOp;
using native_synth::areValidScoreWeights;
using native_synth::CandidateClass;
using native_synth::collectSingleQubitCandidates;
using native_synth::collectTwoQubitBasisCandidates;
using native_synth::collectTwoQubitBasisCandidatesFromMatrix;
using native_synth::collectUnitaryOpsInPreOrder;
using native_synth::decomposeToAxisPair;
using native_synth::decomposeToR;
using native_synth::decomposeToU3;
using native_synth::decomposeToZSXX;
using native_synth::emitSynthesizedSingleQubitFromMatrix;
using native_synth::emitTwoQubitGateSequence;
using native_synth::eulerSequenceForMatrixSynthesis;
using native_synth::getBlockTwoQubitMatrix;
using native_synth::NativeGateKind;
using native_synth::NativeProfileSpec;
using native_synth::resolveNativeGatesSpec;
using native_synth::rewriteXXPlusMinusYYViaRxxRyy;
using native_synth::ScoreWeights;
using native_synth::selectBestCandidate;
using native_synth::SingleQubitEmitterSpec;
using native_synth::SingleQubitMode;
using native_synth::SingleQubitRewritePlan;
using native_synth::SingleQubitRewriteStrategy;
using native_synth::SynthesisCandidate;
using native_synth::TwoQubitWindowConsolidator;
using native_synth::usesCxEntangler;
using native_synth::usesCzEntangler;
using native_synth::xxPlusMinusYyRzzRewriteScoringMetrics;

namespace {

/// Adjacent single-qubit unitaries on one wire considered for fusion.
struct OneQubitRun {
  llvm::SmallVector<UnitaryOpInterface, 4> ops;
};

} // namespace

/// If profitable, replace the run with one synthesized single-qubit op.
static bool maybeFuseRun(IRRewriter& rewriter, OneQubitRun& run,
                         const NativeProfileSpec& spec) {
  Eigen::Matrix2cd fused = Eigen::Matrix2cd::Identity();
  for (UnitaryOpInterface u : run.ops) {
    Eigen::Matrix2cd m;
    if (!u.getUnitaryMatrix2x2(m)) {
      return false;
    }
    fused = m * fused;
  }

  const bool anyNonNative = llvm::any_of(run.ops, [&](UnitaryOpInterface u) {
    return !allowsSingleQubitOp(u, spec);
  });

  assert(!spec.singleQubitEmitters.empty() && "expected at least one emitter");

  constexpr auto kInvalidLen = std::numeric_limits<std::size_t>::max();
  const SingleQubitEmitterSpec* bestEmitter = nullptr;
  std::size_t bestLen = kInvalidLen;
  std::optional<decomposition::QubitGateSequence> bestEuler;
  for (const auto& emitter : spec.singleQubitEmitters) {
    std::size_t len = 0;
    std::optional<decomposition::QubitGateSequence> euler;
    if (emitter.mode == SingleQubitMode::U3) {
      len = 1;
    } else {
      euler = eulerSequenceForMatrixSynthesis(fused, emitter);
      if (!euler) {
        continue;
      }
      len = euler->gates.size();
    }
    if (bestEmitter == nullptr || len < bestLen) {
      bestLen = len;
      bestEmitter = &emitter;
      bestEuler = std::move(euler);
    }
  }
  if (bestEmitter == nullptr) {
    return false;
  }

  // Fully native runs: fuse only if some emitter strictly shortens the chain.
  if (!anyNonNative && bestLen >= run.ops.size()) {
    return false;
  }

  Operation* firstOp = run.ops.front().getOperation();
  const Value inQubit = run.ops.front().getInputQubit(0);
  const Value outQubit = run.ops.back().getOutputQubit(0);

  rewriter.setInsertionPoint(firstOp);
  Value replacement;
  if (bestEmitter->mode == SingleQubitMode::U3) {
    replacement = emitSynthesizedSingleQubitFromMatrix(
        rewriter, firstOp->getLoc(), inQubit, fused, *bestEmitter);
  } else {
    assert(bestEuler.has_value());
    replacement = emitSynthesizedSingleQubitFromMatrix(
        rewriter, firstOp->getLoc(), inQubit, fused, *bestEmitter, &*bestEuler);
  }
  if (!replacement) {
    return false;
  }
  rewriter.replaceAllUsesWith(outQubit, replacement);
  for (auto& op : std::ranges::reverse_view(run.ops)) {
    Operation* toErase = op.getOperation();
    rewriter.eraseOp(toErase);
  }
  return true;
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
  if (llvm::isa_and_present<CtrlOp>(op->getParentOp())) {
    return {};
  }
  Eigen::Matrix2cd matrix;
  if (!unitary.getUnitaryMatrix2x2(matrix)) {
    return {};
  }
  return unitary;
}

/// Dispatch `op`'s direct (non-matrix) single-qubit lowering to the
/// `decomposeTo*` helper for `emitter.mode`. Returns the output qubit value
/// or a null `Value` if no direct rule applies for this op.
static Value
applyDirectSingleQubitLowering(IRRewriter& rewriter, Operation* op, Value in,
                               const SingleQubitEmitterSpec& emitter) {
  switch (emitter.mode) {
  case SingleQubitMode::ZSXX:
    return decomposeToZSXX(rewriter, op, in, emitter.supportsDirectRx);
  case SingleQubitMode::U3:
    return decomposeToU3(rewriter, op, in);
  case SingleQubitMode::R:
    return decomposeToR(rewriter, op, in);
  case SingleQubitMode::AxisPair:
    return decomposeToAxisPair(rewriter, op, in, emitter.axisPair);
  }
  llvm_unreachable("unknown SingleQubitMode");
}

namespace {

/// Lowers unitary QCO ops to a comma-separated native gate menu (single-qubit
/// fuse, two-qubit windows, synthesis sweeps, seam single-qubit fuse, `rz`
/// through `ctrl` controls, another single-qubit fuse, optional cleanup sweeps.
struct NativeGateSynthesisPass
    : impl::NativeGateSynthesisPassBase<NativeGateSynthesisPass> {
  /// Default-construct the pass with the TableGen-generated option defaults.
  NativeGateSynthesisPass() = default;

  /// Construct the pass from the TableGen-generated options struct (forwards
  /// all option values into the base class).
  explicit NativeGateSynthesisPass(
      const NativeGateSynthesisPassOptions& options)
      : NativeGateSynthesisPassBase(options) {}

  /// Construct the pass from the public `NativeGateSynthesisOptions` struct
  /// used by pipeline code that cannot include the TableGen-generated header.
  explicit NativeGateSynthesisPass(const NativeGateSynthesisOptions& options) {
    nativeGates = options.nativeGates;
    scoreWeightTwoQ = options.scoreWeightTwoQ;
    scoreWeightOneQ = options.scoreWeightOneQ;
    scoreWeightDepth = options.scoreWeightDepth;
  }

protected:
  /// Top-level pass entry point. Validates the score weights and native-gate
  /// menu, then drives the staged rewrite pipeline: one-qubit run fusion,
  /// two-qubit window consolidation, synthesis sweeps until the single-qubit
  /// surface is native, seam cleanup, `rz`-through-`ctrl` folding, and a
  /// final fusion pass. Fails the pass on invalid input or non-convergence.
  void runOnOperation() override {
    const ScoreWeights weights{.twoQ = scoreWeightTwoQ,
                               .oneQ = scoreWeightOneQ,
                               .depth = scoreWeightDepth};
    if (!areValidScoreWeights(weights)) {
      getOperation().emitError()
          << "invalid native synthesis score weights (twoq=" << scoreWeightTwoQ
          << ", oneq=" << scoreWeightOneQ << ", depth=" << scoreWeightDepth
          << ")";
      signalPassFailure();
      return;
    }

    // Empty native-gates string: no-op.
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
    const auto& spec = *specOpt;

    IRRewriter rewriter(&getContext());

    fuseOneQubitRuns(rewriter, spec);
    consolidateTwoQubitBlocks(rewriter, spec, weights);
    // Two-qubit lowering can emit off-menu single-qubit ops (e.g. `rx`/`ry`);
    // repeat until clean or hit the sweep cap before seam / `rz` cleanup.
    constexpr unsigned kMaxSynthesisSweeps = 4;
    for (unsigned i = 0; i < kMaxSynthesisSweeps; ++i) {
      if (failed(synthesizeRemainingOps(rewriter, spec, weights))) {
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
    fuseOneQubitRuns(rewriter, spec);
    // Fuse `rz` through control wires of `ctrl` (diagonal control phase).
    fuseRzAcrossCtrlControls(rewriter);
    fuseOneQubitRuns(rewriter, spec);
    // Re-check full menu (single-qubit ops, native `ctrl`, allowed bare `rzz`).
    constexpr unsigned kPostMenuCleanupSweeps = 4;
    unsigned postMenuSweepsRemaining = kPostMenuCleanupSweeps;
    while (hasNonNativeMenuOps(spec) && postMenuSweepsRemaining-- > 0) {
      if (failed(synthesizeRemainingOps(rewriter, spec, weights))) {
        signalPassFailure();
        return;
      }
      fuseOneQubitRuns(rewriter, spec);
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
  static bool ctrlMatchesNativeMenu(CtrlOp ctrl,
                                    const NativeProfileSpec& spec) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    Operation* body = ctrl.getBodyUnitary().getOperation();
    const bool hasCX = llvm::isa<XOp>(body);
    const bool hasCZ = llvm::isa<ZOp>(body);
    if (!hasCX && !hasCZ) {
      return false;
    }
    return (usesCxEntangler(spec) && hasCX) || (usesCzEntangler(spec) && hasCZ);
  }

  /// Bare two-qubit on-menu: `rzz` when the profile allows it.
  static bool bareTwoQubitMatchesNativeMenu(Operation* op,
                                            const NativeProfileSpec& spec) {
    return llvm::isa<RZZOp>(op) && spec.allowRzz &&
           spec.allowedGates.contains(NativeGateKind::Rzz);
  }

  /// True if any unitary is outside `spec` (single-qubit, `ctrl`, or bare
  /// `rzz`).
  bool hasNonNativeMenuOps(const NativeProfileSpec& spec) {
    const mlir::WalkResult walkResult =
        getOperation()->walk([&](Operation* op) {
          if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
            return mlir::WalkResult::advance();
          }
          if (llvm::isa_and_present<CtrlOp>(op->getParentOp())) {
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
  bool hasNonNativeSingleQubitOps(const NativeProfileSpec& spec) {
    const mlir::WalkResult walkResult =
        getOperation()->walk([&](Operation* op) {
          if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
            return mlir::WalkResult::advance();
          }
          if (llvm::isa_and_present<CtrlOp>(op->getParentOp())) {
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
  void fuseOneQubitRuns(IRRewriter& rewriter, const NativeProfileSpec& spec) {
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
      (void)maybeFuseRun(rewriter, run, spec);
    }
  }

  /// If `rz1` can reach another `rz` through at least one `ctrl` control hop,
  /// merge angles into `rz1` and erase the partner.
  ///
  /// `Rz` commutes with a `ctrl` operation acting on the same wire when the
  /// wire is a *control* line (controls only diagonalize the computational
  /// basis and are invariant under Z-rotations). We walk the def-use chain
  /// forward from `rz1`'s output, hopping through `ctrl`s where the wire is
  /// used as a control, and fold into the next `rz` we find. The `hops == 0`
  /// guard intentionally rejects two adjacent `rz`s with nothing in between
  /// -- that case is handled by `fuseOneQubitRuns` above.
  static bool tryFuseRzForwardThroughCtrls(IRRewriter& rewriter, RZOp rz1) {
    Value v = rz1->getResult(0);
    if (!llvm::isa<qco::QubitType>(v.getType())) {
      return false;
    }
    RZOp partner;
    unsigned hops = 0;
    while (v.hasOneUse()) {
      Operation* user = *v.getUsers().begin();
      if (auto rz2 = llvm::dyn_cast<RZOp>(user);
          rz2 && rz2->getOperand(0) == v) {
        partner = rz2;
        break;
      }
      auto ctrl = llvm::dyn_cast<CtrlOp>(user);
      if (!ctrl) {
        return false;
      }
      // Only control wires commute through `ctrl` here.
      if (!llvm::is_contained(ctrl.getControlsIn(), v)) {
        return false;
      }
      v = ctrl.getOutputForInput(v);
      ++hops;
    }
    if (!partner || hops == 0) {
      return false;
    }

    // Fold angles; use a scalar constant when both inputs are constant.
    const Location loc = rz1.getLoc();
    const Value theta1 = rz1.getTheta();
    const Value theta2 = partner.getTheta();
    const auto c1 = mlir::utils::valueToDouble(theta1);
    const auto c2 = mlir::utils::valueToDouble(theta2);
    rewriter.setInsertionPoint(rz1);
    Value newTheta;
    if (c1.has_value() && c2.has_value()) {
      newTheta = mlir::utils::constantFromScalar(rewriter, loc, *c1 + *c2);
    } else {
      newTheta = arith::AddFOp::create(rewriter, loc, theta1, theta2);
    }
    rz1.getThetaMutable().assign(newTheta);
    rewriter.replaceOp(partner, partner->getOperand(0));
    return true;
  }

  /// Fixpoint: merge `rz` through `ctrl` control chains into the next `rz`.
  void fuseRzAcrossCtrlControls(IRRewriter& rewriter) {
    bool changed = true;
    while (changed) {
      changed = false;
      llvm::SmallVector<RZOp, 32> rzOps;
      getOperation()->walk([&](RZOp rz) { rzOps.push_back(rz); });
      for (RZOp rz : rzOps) {
        if (rz->getBlock() == nullptr) {
          continue;
        }
        if (tryFuseRzForwardThroughCtrls(rewriter, rz)) {
          changed = true;
        }
      }
    }
  }

  /// Two-qubit windows with absorbed single-qubit ops: replace when a cheaper
  /// native sequence exists.
  void consolidateTwoQubitBlocks(IRRewriter& rewriter,
                                 const NativeProfileSpec& spec,
                                 const ScoreWeights& weights) {
    llvm::SmallVector<Operation*, 32> ops;
    collectUnitaryOpsInPreOrder(getOperation(), ops);
    TwoQubitWindowConsolidator consolidator;
    for (Operation* op : ops) {
      consolidator.process(op, spec);
    }
    consolidator.materialize(rewriter, spec, weights);
  }

  /// Lower one single-qubit rewrite plan; null `Value` on failure.
  static Value emitSingleQCandidate(IRRewriter& rewriter, Operation* op,
                                    UnitaryOpInterface unitary,
                                    const SingleQubitRewritePlan& plan) {
    const Value in = unitary.getInputQubit(0);
    if (plan.strategy == SingleQubitRewriteStrategy::Direct) {
      return applyDirectSingleQubitLowering(rewriter, op, in, plan.emitter);
    }
    Eigen::Matrix2cd matrix;
    if (!unitary.isSingleQubit() || !unitary.getUnitaryMatrix2x2(matrix)) {
      return {};
    }
    return emitSynthesizedSingleQubitFromMatrix(rewriter, op->getLoc(), in,
                                                matrix, plan.emitter);
  }

  /// One synthesis sweep over the whole function: rewrite every remaining
  /// off-menu unitary by dispatching to `rewriteSingleQubit` /
  /// `rewriteControlled` / `rewriteTwoQubit`. Returns `failure()` as soon as
  /// any op cannot be lowered to the native menu. Safe to call repeatedly;
  /// `runOnOperation` iterates until convergence.
  LogicalResult synthesizeRemainingOps(IRRewriter& rewriter,
                                       const NativeProfileSpec& spec,
                                       const ScoreWeights& weights) {
    llvm::SmallVector<Operation*, 32> ops;
    collectUnitaryOpsInPreOrder(getOperation(), ops);

    for (Operation* op : ops) {
      // Pointers were collected before this loop.
      if (op->getBlock() == nullptr) {
        continue;
      }
      // Inner `CtrlOp` bodies are handled on the `CtrlOp` itself.
      if (llvm::isa_and_present<CtrlOp>(op->getParentOp())) {
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
          if (failed(
                  rewriteSingleQubit(rewriter, op, unitary, spec, weights))) {
            return failure();
          }
        }
        continue;
      }

      if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
        if (failed(rewriteControlled(rewriter, ctrl, spec, weights))) {
          return failure();
        }
        continue;
      }

      if (unitary.isTwoQubit()) {
        if (failed(rewriteTwoQubit(rewriter, op, unitary, spec, weights))) {
          return failure();
        }
        continue;
      }
    }
    return success();
  }

  /// Lower one off-menu single-qubit `op`: enumerate all valid rewrite
  /// candidates for the active native profile, pick the best by `weights`,
  /// emit it, and replace `op`.
  static LogicalResult rewriteSingleQubit(IRRewriter& rewriter, Operation* op,
                                          UnitaryOpInterface unitary,
                                          const NativeProfileSpec& spec,
                                          const ScoreWeights& weights) {
    rewriter.setInsertionPoint(op);
    const auto candidates = collectSingleQubitCandidates(unitary, spec);
    const auto* best = selectBestCandidate(llvm::ArrayRef(candidates), weights);
    const Value replaced =
        best != nullptr
            ? emitSingleQCandidate(rewriter, op, unitary, best->payload)
            : Value{};
    if (!replaced) {
      op->emitError("single-qubit operation not in selected native profile");
      return failure();
    }
    rewriter.replaceOp(op, replaced);
    return success();
  }

  /// Lower a single-control, single-target `CtrlOp` to the native profile.
  /// Fast-path: already-native `CX`/`CZ` are kept as-is. Otherwise, lift the
  /// controlled op to its 4x4 matrix (with SU(4) normalization), run the
  /// Weyl-based basis-decomposer search, and emit the best candidate.
  static LogicalResult rewriteControlled(IRRewriter& rewriter, CtrlOp ctrl,
                                         const NativeProfileSpec& spec,
                                         const ScoreWeights& weights) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      ctrl.emitError("native synthesis currently only supports 1-control "
                     "1-target controlled gates");
      return failure();
    }
    auto* body = ctrl.getBodyUnitary().getOperation();
    const bool hasCX = llvm::isa<XOp>(body);
    const bool hasCZ = llvm::isa<ZOp>(body);
    if ((usesCxEntangler(spec) && hasCX) || (usesCzEntangler(spec) && hasCZ)) {
      return success();
    }
    // Otherwise treat as a generic `4×4` (Weyl + basis decomposer + scorer).
    Eigen::Matrix4cd matrix;
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
    native_synth::normalizeToSU4(matrix); // SU(4) convention for Weyl

    const auto candidates =
        collectTwoQubitBasisCandidatesFromMatrix(matrix, spec);
    if (const auto* best =
            selectBestCandidate(llvm::ArrayRef(candidates), weights)) {
      rewriter.setInsertionPoint(ctrl);
      if (succeeded(emitTwoQubitGateSequence(
              rewriter, ctrl.getOperation(), ctrl.getInputControl(0),
              ctrl.getInputTarget(0), best->payload.sequence))) {
        return success();
      }
    }
    ctrl.emitError("controlled gate not allowed by selected profile");
    return failure();
  }

  /// Lower an off-menu generic two-qubit op (`RZZ`, `XXPlusYY`, `XXMinusYY`,
  /// or any arbitrary 4x4 unitary). Handles the `Rzz`-native fast path and
  /// the `XXPlusMinusYY -> Rzz` specialization first, then falls back to the
  /// Weyl-based basis-decomposer search.
  static LogicalResult rewriteTwoQubit(IRRewriter& rewriter, Operation* op,
                                       UnitaryOpInterface unitary,
                                       const NativeProfileSpec& spec,
                                       const ScoreWeights& weights) {
    if (spec.allowRzz && llvm::isa<RZZOp>(op)) {
      return success();
    }
    if (spec.allowRzz &&
        (llvm::isa<XXPlusYYOp>(op) || llvm::isa<XXMinusYYOp>(op))) {
      llvm::SmallVector<SynthesisCandidate<bool>> candidates;
      candidates.push_back(SynthesisCandidate<bool>{
          .candidateClass = CandidateClass::XxPlusMinusViaRzz,
          .metrics = xxPlusMinusYyRzzRewriteScoringMetrics(),
          .enumerationIndex = 0,
          .payload = true,
      });
      if (selectBestCandidate(llvm::ArrayRef(candidates), weights) != nullptr) {
        rewriter.setInsertionPoint(op);
        if (succeeded(rewriteXXPlusMinusYYViaRxxRyy(rewriter, op))) {
          return success();
        }
      }
    }
    if (!spec.entanglerBases.empty()) {
      const auto candidates = collectTwoQubitBasisCandidates(unitary, spec);
      if (const auto* best =
              selectBestCandidate(llvm::ArrayRef(candidates), weights)) {
        rewriter.setInsertionPoint(op);
        if (succeeded(emitTwoQubitGateSequence(
                rewriter, op, unitary.getInputQubit(0),
                unitary.getInputQubit(1), best->payload.sequence))) {
          return success();
        }
      }
    }
    op->emitError("unsupported two-qubit operation for selected profile");
    return failure();
  }
};

} // namespace

std::unique_ptr<Pass>
createNativeGateSynthesisPass(const NativeGateSynthesisOptions& options) {
  return std::make_unique<NativeGateSynthesisPass>(options);
}

} // namespace mlir::qco
