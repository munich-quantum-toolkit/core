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
#include "mlir/Dialect/QCO/Transforms/Decomposition/NativeGateset.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cassert>
#include <cstddef>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_FUSETWOQUBITUNITARYRUNS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

using decomposition::EulerBasis;
using decomposition::NativeGateset;
using decomposition::populateFuseSingleQubitUnitaryRunsPatterns;
using decomposition::synthesizeUnitary2QWeyl;

namespace {

/** Composed unitary and metadata for a fusable two-qubit run. */
struct FusableTwoQubitRun {
  SmallVector<Operation*, 8> ops; ///< Members in program order.
  Matrix4x4 composed = Matrix4x4::identity();
  unsigned numTwoQ = 0; ///< Number of two-qubit members (entanglers consumed).
  bool hasNonNativeGate = false; ///< Any member off the native gateset.
  Value tailA;                   ///< Current output wires of the run's tail.
  Value tailB;
};

} // namespace

// --- Run membership ------------------------------------------------------- //

/// Whether `op` is nested under a `ctrl`/`inv` body. Such unitaries are handled
/// through their shell op, so the top-level walk skips them.
static bool isExcludedFromTopLevelUnitaryWalk(Operation* op) {
  if (op->getParentOfType<CtrlOp>()) {
    return true;
  }
  return !llvm::isa<InvOp>(op) && op->getParentOfType<InvOp>();
}

/// Whether `op` is a unitary shell the pass may rewrite at top level.
static bool isWalkableUnitaryShell(Operation* op) {
  return !llvm::isa<BarrierOp, GPhaseOp>(op) &&
         !isExcludedFromTopLevelUnitaryWalk(op);
}

/// Builds the constant 4x4 matrix for a two-qubit op (bare or single-target
/// `CtrlOp`). Returns false for a `CtrlOp` that is not
/// single-control/single-target, or an op whose matrix is not known at compile
/// time.
static bool assignTwoQubitOpMatrix(Operation* op, Matrix4x4& matrix) {
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    return llvm::cast<UnitaryOpInterface>(ctrl.getOperation())
        .getUnitaryMatrix4x4(matrix);
  }
  auto unitary = llvm::cast<UnitaryOpInterface>(op);
  assert(unitary.isTwoQubit() &&
         "only two-qubit unitary shells are passed to assignTwoQubitOpMatrix");
  return unitary.getUnitaryMatrix4x4(matrix);
}

/// Whether `unitary` is a single-qubit gate that can join a run.
static bool isOneQubitRunMember(UnitaryOpInterface unitary) {
  if (!unitary || !unitary.isSingleQubit() ||
      !isWalkableUnitaryShell(unitary.getOperation())) {
    return false;
  }
  Matrix2x2 matrix;
  return unitary.getUnitaryMatrix2x2(matrix);
}

/// Whether `unitary` is a two-qubit gate that can join a run.
static bool isTwoQubitRunMember(UnitaryOpInterface unitary) {
  if (!unitary || !unitary.isTwoQubit() ||
      !isWalkableUnitaryShell(unitary.getOperation())) {
    return false;
  }
  Matrix4x4 matrix;
  return assignTwoQubitOpMatrix(unitary.getOperation(), matrix);
}

// --- Wire navigation ------------------------------------------------------ //

/// The sole run-member consumer of `wire`, or a null interface when its unique
/// user cannot join a run. `wire` is single-use by qubit linearity.
static UnitaryOpInterface uniqueUnitaryUser(Value wire) {
  assert(wire.hasOneUse() &&
         "qubit values are single-use, so a run tail has exactly one user");
  auto unitary = llvm::dyn_cast<UnitaryOpInterface>(*wire.user_begin());
  if (!unitary) {
    return {};
  }
  if (unitary.isTwoQubit()) {
    return isTwoQubitRunMember(unitary) ? unitary : UnitaryOpInterface{};
  }
  if (unitary.isSingleQubit()) {
    return isOneQubitRunMember(unitary) ? unitary : UnitaryOpInterface{};
  }
  return {};
}

/// Traces `wire` upstream through single-qubit gates to the two-qubit run
/// member terminating the chain, or `nullptr` if the chain is broken.
static Operation* twoQubitGateAtEndOfOneQChain(Value wire) {
  Value cur = wire;
  while (Operation* def = cur.getDefiningOp()) {
    auto unitary = llvm::dyn_cast<UnitaryOpInterface>(def);
    if (!unitary) {
      return nullptr;
    }
    if (unitary.isTwoQubit()) {
      return isTwoQubitRunMember(unitary) ? def : nullptr;
    }
    if (!isOneQubitRunMember(unitary)) {
      return nullptr;
    }
    cur = unitary.getInputQubit(0);
  }
  return nullptr;
}

/// Whether both input wires of `op` come from one earlier two-qubit run, making
/// `op` a continuation of that run rather than a fresh run start.
static bool feedsFromSameTwoQubitRun(UnitaryOpInterface op) {
  const Value in0 = op.getInputQubit(0);
  const Value in1 = op.getInputQubit(1);
  assert(in0.hasOneUse() && in1.hasOneUse() &&
         "qubit values are single-use, so a run member consumes each input "
         "exactly once");
  Operation* gate0 = twoQubitGateAtEndOfOneQChain(in0);
  Operation* gate1 = twoQubitGateAtEndOfOneQChain(in1);
  return gate0 != nullptr && gate0 == gate1;
}

// --- Run scanning --------------------------------------------------------- //

/// Appends a two-qubit gate to `run`, composing its matrix. No-op unless both
/// of `op`'s inputs are the run's current tail wires (in either order), keeping
/// the run confined to a single pair of wires.
static void absorbTwoQubitIntoRun(FusableTwoQubitRun& run,
                                  UnitaryOpInterface op,
                                  const NativeGateset& spec) {
  Matrix4x4 opMatrix;
  [[maybe_unused]] const bool assigned =
      assignTwoQubitOpMatrix(op.getOperation(), opMatrix);
  assert(assigned && "a two-qubit run member always exposes a 4x4 matrix");
  const Value in0 = op.getInputQubit(0);
  const Value in1 = op.getInputQubit(1);
  std::size_t id0 = 0;
  std::size_t id1 = 1;
  if (in0 == run.tailA && in1 == run.tailB) {
    run.tailA = op.getOutputQubit(0);
    run.tailB = op.getOutputQubit(1);
  } else if (in0 == run.tailB && in1 == run.tailA) {
    id0 = 1;
    id1 = 0;
    run.tailA = op.getOutputQubit(1);
    run.tailB = op.getOutputQubit(0);
  } else {
    llvm_unreachable(
        "a unique user of both tail wires connects to both of them");
  }
  run.composed.premultiplyBy(opMatrix.reorderForQubits(id0, id1));
  run.ops.push_back(op.getOperation());
  ++run.numTwoQ;
  run.hasNonNativeGate |= !spec.allowsOp(op.getOperation());
}

/// Appends a single-qubit gate on run wire `wireIndex` (0 = A, 1 = B).
static void absorbOneQubitIntoRun(FusableTwoQubitRun& run,
                                  UnitaryOpInterface op,
                                  const NativeGateset& spec,
                                  unsigned wireIndex) {
  Matrix2x2 raw;
  [[maybe_unused]] const bool assigned = op.getUnitaryMatrix2x2(raw);
  assert(assigned && "a single-qubit run member always exposes a 2x2 matrix");
  run.composed.premultiplyBy(raw.embedInTwoQubit(wireIndex));
  run.ops.push_back(op.getOperation());
  run.hasNonNativeGate |= !spec.allowsOp(op.getOperation());
  (wireIndex == 0 ? run.tailA : run.tailB) = op.getOutputQubit(0);
}

/// Walks forward from `head`, composing the run's matrix and metadata. Absorbs
/// a following two-qubit gate when it keeps both run wires together, otherwise
/// the single-qubit gate first in program order; stops at the first boundary
/// that would split the run's two wires.
static FusableTwoQubitRun scanFusableTwoQubitRun(UnitaryOpInterface head,
                                                 const NativeGateset& spec) {
  FusableTwoQubitRun run;
  [[maybe_unused]] const bool assigned =
      assignTwoQubitOpMatrix(head.getOperation(), run.composed);
  assert(assigned && "a run head is a two-qubit member with a 4x4 matrix");
  run.tailA = head.getOutputQubit(0);
  run.tailB = head.getOutputQubit(1);
  run.ops.push_back(head.getOperation());
  run.numTwoQ = 1;
  run.hasNonNativeGate |= !spec.allowsOp(head.getOperation());

  while (true) {
    UnitaryOpInterface nextOnA = uniqueUnitaryUser(run.tailA);
    UnitaryOpInterface nextOnB = uniqueUnitaryUser(run.tailB);
    const bool sameOp =
        nextOnA && nextOnB && nextOnA.getOperation() == nextOnB.getOperation();

    if (sameOp && nextOnA.isTwoQubit()) {
      absorbTwoQubitIntoRun(run, nextOnA, spec);
      continue;
    }

    const bool aSingle = nextOnA && nextOnA.isSingleQubit() && !sameOp;
    const bool bSingle = nextOnB && nextOnB.isSingleQubit() && !sameOp;
    if (aSingle && bSingle && nextOnA->getBlock() != nextOnB->getBlock()) {
      break;
    }
    if (aSingle && (!bSingle || nextOnA->isBeforeInBlock(nextOnB))) {
      absorbOneQubitIntoRun(run, nextOnA, spec, /*wireIndex=*/0);
      continue;
    }
    if (bSingle) {
      absorbOneQubitIntoRun(run, nextOnB, spec, /*wireIndex=*/1);
      continue;
    }
    break;
  }
  return run;
}

/// Erases all run members, successors first so each is dead when erased.
static void eraseFusableRun(PatternRewriter& rewriter,
                            const FusableTwoQubitRun& run) {
  for (Operation* member : llvm::reverse(run.ops)) {
    rewriter.eraseOp(member);
  }
}

/// Whether any single- or two-qubit unitary (including `ctrl` shells) remains
/// off the native gateset. Used as the pass' convergence check. Gates acting on
/// more than two qubits are out of scope here (a dedicated multi-controlled
/// synthesis pass possibly lowers those) and are left untouched rather than
/// reported.
static bool hasNonNativeOps(Operation* root, const NativeGateset& spec) {
  const WalkResult walkResult = root->walk([&](Operation* op) {
    auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
    if (!unitary || !isWalkableUnitaryShell(op) || unitary.getNumQubits() > 2) {
      return WalkResult::advance();
    }
    return spec.allowsOp(op) ? WalkResult::advance() : WalkResult::interrupt();
  });
  return walkResult.wasInterrupted();
}

namespace {

/// Fuses a maximal two-qubit run into one composed unitary and resynthesizes it
/// to the native gateset when beneficial.
struct FuseTwoQubitUnitaryRunsPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  FuseTwoQubitUnitaryRunsPattern(MLIRContext* ctx, NativeGateset specIn)
      : OpInterfaceRewritePattern(ctx), spec(std::move(specIn)) {}

  NativeGateset spec;

  /// Whether `op` anchors a run: a two-qubit run member whose two wires are not
  /// both fed by the same earlier run (which would make it a continuation).
  static bool isRunStart(UnitaryOpInterface op) {
    return isTwoQubitRunMember(op) && !feedsFromSameTwoQubitRun(op);
  }

  /// Fuses the run anchored at `op` if it contains an off-gateset gate or Weyl
  /// resynthesis uses fewer entanglers than the run's two-qubit members.
  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    if (!isRunStart(op)) {
      return failure();
    }

    FusableTwoQubitRun run = scanFusableTwoQubitRun(op, spec);
    if (run.ops.size() < 2) {
      return failure();
    }

    const auto native = spec.decomposeTarget(run.composed);
    if (!native ||
        (!run.hasNonNativeGate && native->numBasisUses >= run.numTwoQ)) {
      return failure();
    }

    auto firstOp = llvm::cast<UnitaryOpInterface>(run.ops.front());
    rewriter.setInsertionPoint(firstOp);
    Value newA;
    Value newB;
    if (failed(synthesizeUnitary2QWeyl(
            rewriter, firstOp.getLoc(), firstOp.getInputQubit(0),
            firstOp.getInputQubit(1), run.composed, spec, newA, newB))) {
      firstOp->emitError("failed to emit synthesized two-qubit gate sequence");
      return failure();
    }
    rewriter.replaceAllUsesWith(run.tailA, newA);
    rewriter.replaceAllUsesWith(run.tailB, newB);
    eraseFusableRun(rewriter, run);
    return success();
  }
};

/// Lowers a single off-gateset two-qubit op (bare or single-target `CtrlOp`) to
/// the native entangler plus native single-qubit factors via Weyl synthesis.
/// Native two-qubit ops and fusable runs are left to
/// @ref FuseTwoQubitUnitaryRunsPattern.
struct LowerTwoQubitOpPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  LowerTwoQubitOpPattern(MLIRContext* ctx, NativeGateset specIn)
      : OpInterfaceRewritePattern(ctx), spec(std::move(specIn)) {}

  NativeGateset spec;

  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    Operation* raw = op.getOperation();
    if (!isWalkableUnitaryShell(raw) || spec.allowsOp(raw)) {
      return failure();
    }
    Matrix4x4 matrix;
    if (!assignTwoQubitOpMatrix(raw, matrix)) {
      return failure();
    }

    Value in0;
    Value in1;
    if (auto ctrl = llvm::dyn_cast<CtrlOp>(raw)) {
      in0 = ctrl.getInputControl(0);
      in1 = ctrl.getInputTarget(0);
    } else {
      in0 = op.getInputQubit(0);
      in1 = op.getInputQubit(1);
    }

    rewriter.setInsertionPoint(raw);
    Value out0;
    Value out1;
    if (failed(synthesizeUnitary2QWeyl(rewriter, raw->getLoc(), in0, in1,
                                       matrix, spec, out0, out1))) {
      return failure();
    }
    rewriter.replaceOp(raw, ValueRange{out0, out1});
    return success();
  }
};

} // namespace

/// Fuses single-qubit runs (and lowers lone off-gateset single-qubit ops) by
/// reusing the `fuse-single-qubit-unitary-runs` rewrite. `qco.ctrl` bodies are
/// skipped so the `X`/`Z` bodies of native entanglers are preserved.
static LogicalResult fuseSingleQubitRuns(ModuleOp module,
                                         const EulerBasis basis) {
  RewritePatternSet patterns(module.getContext());
  populateFuseSingleQubitUnitaryRunsPatterns(patterns, basis,
                                             /*skipControlledBodies=*/true);
  return applyPatternsGreedily(module, std::move(patterns));
}

/// Fuses two-qubit runs, then lowers any remaining off-gateset two-qubit ops.
static LogicalResult fuseAndLowerTwoQubitOps(ModuleOp module,
                                             const NativeGateset& spec) {
  MLIRContext* ctx = module.getContext();
  {
    RewritePatternSet runPatterns(ctx);
    runPatterns.add<FuseTwoQubitUnitaryRunsPattern>(ctx, spec);
    if (failed(applyPatternsGreedily(module, std::move(runPatterns)))) {
      return failure();
    }
  }
  RewritePatternSet lowerPatterns(ctx);
  lowerPatterns.add<LowerTwoQubitOpPattern>(ctx, spec);
  return applyPatternsGreedily(module, std::move(lowerPatterns));
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
    const auto spec = NativeGateset::parse(nativeGates);
    if (!spec) {
      getOperation().emitError() << "unsupported native gateset (native-gates='"
                                 << nativeGates << "')";
      signalPassFailure();
      return;
    }
    const EulerBasis basis = *spec->eulerBasis;
    ModuleOp module = getOperation();

    // 1. Fuse single-qubit runs (also lowers lone off-gateset single-qubit
    // ops).
    // 2. Fuse two-qubit runs and lower remaining off-gateset two-qubit ops.
    // 3. Fuse the single-qubit seams introduced by two-qubit synthesis.
    if (failed(fuseSingleQubitRuns(module, basis)) ||
        failed(fuseAndLowerTwoQubitOps(module, *spec)) ||
        failed(fuseSingleQubitRuns(module, basis))) {
      signalPassFailure();
      return;
    }

    if (hasNonNativeOps(module, *spec)) {
      module.emitError() << "native gate synthesis: operations remain outside "
                            "the native gateset (native-gates='"
                         << nativeGates << "')";
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco
