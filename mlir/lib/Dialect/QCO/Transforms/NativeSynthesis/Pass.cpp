/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/NativeSpec.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/PassTwoQubitWindows.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Policy.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/SingleQubit.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/TwoQubit.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Utils.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>

#include <cstddef>
#include <memory>
#include <optional>

namespace mlir::qco {
#define GEN_PASS_DEF_NATIVEGATESYNTHESISPASS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"
} // namespace mlir::qco

namespace mlir::qco {

using native_synth::allowsSingleQubitOp;
using native_synth::canDirectlyDecomposeToAxisPair;
using native_synth::canDirectlyDecomposeToR;
using native_synth::canDirectlyDecomposeToU3;
using native_synth::canDirectlyDecomposeToZSXX;
using native_synth::collectUnitaryOpsInPreOrder;
using native_synth::decomposeToAxisPair;
using native_synth::decomposeToR;
using native_synth::decomposeToU3;
using native_synth::decomposeToZSXX;
using native_synth::emitSingleQubitMatrix;
using native_synth::emitterEulerBasis;
using native_synth::emitTwoQubitNative;
using native_synth::getBlockTwoQubitMatrix;
using native_synth::NativeGateKind;
using native_synth::NativeProfileSpec;
using native_synth::resolveNativeGatesSpec;
using native_synth::rewriteXXPlusMinusYYViaRzz;
using native_synth::SingleQubitEmitterSpec;
using native_synth::SingleQubitMode;
using native_synth::TwoQubitWindowConsolidator;
using native_synth::usesCxEntangler;
using native_synth::usesCzEntangler;

namespace {

/// Adjacent single-qubit unitaries on one wire considered for fusion.
struct OneQubitRun {
  llvm::SmallVector<UnitaryOpInterface, 4> ops;
};

} // namespace

/// If profitable, replace the run with one synthesized single-qubit op in
/// `basis` (mirrors `FuseSingleQubitUnitaryRuns`). Fuses when any op is
/// off-menu or when Euler resynthesis strictly shortens the run.
static bool maybeFuseRun(IRRewriter& rewriter, OneQubitRun& run,
                         const decomposition::EulerBasis basis,
                         const NativeProfileSpec& spec) {
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

/// Whether `emitter` can lower the single-qubit `op` directly (used for ops
/// with non-constant angles, which have no constant `2×2` matrix).
static bool emitterHasDirectLowering(Operation* op,
                                     const SingleQubitEmitterSpec& emitter) {
  switch (emitter.mode) {
  case SingleQubitMode::ZSXX:
    return canDirectlyDecomposeToZSXX(op, emitter.supportsDirectRx);
  case SingleQubitMode::U3:
    return canDirectlyDecomposeToU3(op);
  case SingleQubitMode::R:
    return canDirectlyDecomposeToR(op);
  case SingleQubitMode::AxisPair:
    return canDirectlyDecomposeToAxisPair(op, emitter.axisPair);
  }
  return false;
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

/// Lowers unitary QCO ops to a comma-separated native gate menu using a
/// deterministic, matrix-driven synthesizer: single-qubit fuse, two-qubit
/// window consolidation, synthesis sweeps, seam single-qubit fuse, and
/// optional cleanup sweeps.
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
  }

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
    auto specOpt = resolveNativeGatesSpec(nativeGates);
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
        emitterEulerBasis(spec.singleQubitEmitters.front());

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
  static bool ctrlMatchesNativeMenu(CtrlOp ctrl,
                                    const NativeProfileSpec& spec) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    Operation* body = ctrl.getBodyUnitary(0).getOperation();
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
  bool hasNonNativeSingleQubitOps(const NativeProfileSpec& spec) {
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
  void fuseOneQubitRuns(IRRewriter& rewriter, const NativeProfileSpec& spec,
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
  LogicalResult consolidateTwoQubitBlocks(IRRewriter& rewriter,
                                          const NativeProfileSpec& spec) {
    llvm::SmallVector<Operation*, 32> ops;
    collectUnitaryOpsInPreOrder(getOperation(), ops);
    TwoQubitWindowConsolidator consolidator;
    for (Operation* op : ops) {
      consolidator.process(op, spec);
    }
    return consolidator.materialize(rewriter, spec);
  }

  /// One synthesis sweep over the whole function: rewrite every remaining
  /// off-menu unitary by dispatching to `rewriteSingleQubit` /
  /// `rewriteControlled` / `rewriteTwoQubit`. Returns `failure()` as soon as
  /// any op cannot be lowered to the native menu. Safe to call repeatedly;
  /// `runOnOperation` iterates until convergence.
  LogicalResult synthesizeRemainingOps(IRRewriter& rewriter,
                                       const NativeProfileSpec& spec,
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
          if (failed(rewriteSingleQubit(rewriter, op, unitary, spec, basis))) {
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

  /// Lower one off-menu single-qubit `op`. Constant unitaries use the
  /// matrix-driven Euler synthesizer in `basis`; ops with non-constant angles
  /// fall back to the symbolic `decomposeTo*` lowering of the first emitter
  /// that handles them.
  static LogicalResult
  rewriteSingleQubit(IRRewriter& rewriter, Operation* op,
                     UnitaryOpInterface unitary, const NativeProfileSpec& spec,
                     const decomposition::EulerBasis basis) {
    rewriter.setInsertionPoint(op);
    const Value in = unitary.getInputQubit(0);
    Matrix2x2 matrix;
    if (unitary.isSingleQubit() && unitary.getUnitaryMatrix2x2(matrix)) {
      const Value replaced =
          emitSingleQubitMatrix(rewriter, op->getLoc(), in, matrix, basis);
      rewriter.replaceOp(op, replaced);
      return success();
    }
    for (const auto& emitter : spec.singleQubitEmitters) {
      if (!emitterHasDirectLowering(op, emitter)) {
        continue;
      }
      if (const Value replaced =
              applyDirectSingleQubitLowering(rewriter, op, in, emitter)) {
        rewriter.replaceOp(op, replaced);
        return success();
      }
    }
    op->emitError("single-qubit operation not in selected native profile");
    return failure();
  }

  /// Lower a single-control, single-target `CtrlOp` to the native profile.
  /// Fast-path: already-native `CX`/`CZ` are kept as-is. Otherwise, lift the
  /// controlled op to its 4x4 matrix and run the deterministic two-qubit
  /// synthesizer.
  static LogicalResult rewriteControlled(IRRewriter& rewriter, CtrlOp ctrl,
                                         const NativeProfileSpec& spec) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      ctrl.emitError("native synthesis currently only supports 1-control "
                     "1-target controlled gates");
      return failure();
    }
    auto* body = ctrl.getBodyUnitary(0).getOperation();
    const bool hasCX = llvm::isa<XOp>(body);
    const bool hasCZ = llvm::isa<ZOp>(body);
    if ((usesCxEntangler(spec) && hasCX) || (usesCzEntangler(spec) && hasCZ)) {
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
    if (failed(emitTwoQubitNative(
            rewriter, ctrl.getLoc(), ctrl.getInputControl(0),
            ctrl.getInputTarget(0), matrix, spec, out0, out1))) {
      ctrl.emitError("controlled gate not allowed by selected profile");
      return failure();
    }
    rewriter.replaceOp(ctrl, ValueRange{out0, out1});
    return success();
  }

  /// Lower an off-menu generic two-qubit op (`RZZ`, `XXPlusYY`, `XXMinusYY`,
  /// or any arbitrary 4x4 unitary). Handles the `Rzz`-native fast path; for
  /// `XXPlusYY` / `XXMinusYY` with `rzz` on the menu, uses the dedicated
  /// `XX±YY -> Rzz` rewrite. All other two-qubit unitaries go through the
  /// deterministic KAK synthesizer.
  static LogicalResult rewriteTwoQubit(IRRewriter& rewriter, Operation* op,
                                       UnitaryOpInterface unitary,
                                       const NativeProfileSpec& spec) {
    if (spec.allowRzz && llvm::isa<RZZOp>(op)) {
      return success();
    }
    if (spec.allowRzz &&
        (llvm::isa<XXPlusYYOp>(op) || llvm::isa<XXMinusYYOp>(op))) {
      rewriter.setInsertionPoint(op);
      if (succeeded(rewriteXXPlusMinusYYViaRzz(rewriter, op))) {
        return success();
      }
      // Fall through to entangler-based synthesis when the dedicated rewrite
      // could not be applied (e.g. no entangler-free realization).
    }
    Matrix4x4 matrix;
    if (!getBlockTwoQubitMatrix(op, matrix)) {
      op->emitError("unsupported two-qubit operation for selected profile");
      return failure();
    }
    rewriter.setInsertionPoint(op);
    Value out0;
    Value out1;
    if (failed(emitTwoQubitNative(
            rewriter, op->getLoc(), unitary.getInputQubit(0),
            unitary.getInputQubit(1), matrix, spec, out0, out1))) {
      op->emitError("unsupported two-qubit operation for selected profile");
      return failure();
    }
    rewriter.replaceOp(op, ValueRange{out0, out1});
    return success();
  }
};

} // namespace

std::unique_ptr<Pass>
createNativeGateSynthesisPass(const NativeGateSynthesisOptions& options) {
  return std::make_unique<NativeGateSynthesisPass>(options);
}

} // namespace mlir::qco
