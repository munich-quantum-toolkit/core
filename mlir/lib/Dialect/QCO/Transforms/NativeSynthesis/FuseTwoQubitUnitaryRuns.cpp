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
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

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

/// Skips unitaries nested under `ctrl`/`inv` bodies (handled on the shell op).
static bool isExcludedFromTopLevelUnitaryWalk(Operation* op) {
  if (op->getParentOfType<CtrlOp>()) {
    return true;
  }
  return !llvm::isa<InvOp>(op) && op->getParentOfType<InvOp>();
}

static bool isWalkableUnitaryShell(Operation* op) {
  return !llvm::isa<BarrierOp, GPhaseOp>(op) &&
         !isExcludedFromTopLevelUnitaryWalk(op);
}

/// Builds the constant 4x4 matrix for a two-qubit op (bare or single-target
/// `CtrlOp`). Returns false for barriers, global phase, non-two-qubit ops, and
/// ops whose matrix is not known at compile time.
static bool assignTwoQubitOpMatrix(Operation* op, Matrix4x4& matrix) {
  if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
    return false;
  }
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    return llvm::cast<UnitaryOpInterface>(ctrl.getOperation())
        .getUnitaryMatrix4x4(matrix);
  }
  auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
  if (!unitary || !unitary.isTwoQubit()) {
    return false;
  }
  return unitary.getUnitaryMatrix4x4(matrix);
}

static bool isOneQubitWindowMember(UnitaryOpInterface unitary) {
  if (!unitary || !unitary.isSingleQubit() ||
      !isWalkableUnitaryShell(unitary.getOperation())) {
    return false;
  }
  Matrix2x2 matrix;
  return unitary.getUnitaryMatrix2x2(matrix);
}

static bool isTwoQubitRunMember(UnitaryOpInterface unitary) {
  if (!unitary || !unitary.isTwoQubit() ||
      !isWalkableUnitaryShell(unitary.getOperation())) {
    return false;
  }
  Matrix4x4 matrix;
  return assignTwoQubitOpMatrix(unitary.getOperation(), matrix);
}

static UnitaryOpInterface uniqueUnitaryUser(Value wire) {
  if (!wire.hasOneUse()) {
    return {};
  }
  auto unitary = llvm::dyn_cast<UnitaryOpInterface>(*wire.user_begin());
  if (!unitary) {
    return {};
  }
  if (unitary.isTwoQubit()) {
    return isTwoQubitRunMember(unitary) ? unitary : UnitaryOpInterface{};
  }
  if (unitary.isSingleQubit()) {
    return isOneQubitWindowMember(unitary) ? unitary : UnitaryOpInterface{};
  }
  return {};
}

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
    if (!isOneQubitWindowMember(unitary)) {
      return nullptr;
    }
    cur = unitary.getInputQubit(0);
  }
  return nullptr;
}

static bool feedsFromSameTwoQubitWindow(UnitaryOpInterface op) {
  const Value in0 = op.getInputQubit(0);
  const Value in1 = op.getInputQubit(1);
  if (!in0.hasOneUse() || !in1.hasOneUse()) {
    return false;
  }
  Operation* gate0 = twoQubitGateAtEndOfOneQChain(in0);
  Operation* gate1 = twoQubitGateAtEndOfOneQChain(in1);
  return gate0 != nullptr && gate0 == gate1;
}

namespace {

struct FusableTwoQubitRun {
  SmallVector<Operation*, 8> ops;
  Matrix4x4 composed = Matrix4x4::identity();
  unsigned numTwoQ = 0;
  bool anyNonNative = false;
  Value tailA;
  Value tailB;
};

} // namespace

static void absorbTwoQubitIntoRun(FusableTwoQubitRun& run,
                                  UnitaryOpInterface op,
                                  const NativeGateset& spec) {
  Matrix4x4 opMatrix;
  if (!assignTwoQubitOpMatrix(op.getOperation(), opMatrix)) {
    return;
  }
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
    return;
  }
  run.composed.premultiplyBy(opMatrix.reorderForQubits(id0, id1));
  run.ops.push_back(op.getOperation());
  ++run.numTwoQ;
  run.anyNonNative |= !spec.allowsOp(op.getOperation());
}

static void absorbOneQubitIntoRun(FusableTwoQubitRun& run,
                                  UnitaryOpInterface op,
                                  const NativeGateset& spec,
                                  unsigned wireIndex) {
  Matrix2x2 raw;
  if (!op.getUnitaryMatrix2x2(raw)) {
    return;
  }
  run.composed.premultiplyBy(raw.embedInTwoQubit(wireIndex));
  run.ops.push_back(op.getOperation());
  run.anyNonNative |= !spec.allowsOp(op.getOperation());
  (wireIndex == 0 ? run.tailA : run.tailB) = op.getOutputQubit(0);
}

static FusableTwoQubitRun scanFusableTwoQubitRun(UnitaryOpInterface head,
                                                 const NativeGateset& spec) {
  FusableTwoQubitRun run;
  if (!assignTwoQubitOpMatrix(head.getOperation(), run.composed)) {
    return run;
  }
  run.tailA = head.getOutputQubit(0);
  run.tailB = head.getOutputQubit(1);
  run.ops.push_back(head.getOperation());
  run.numTwoQ = 1;
  run.anyNonNative |= !spec.allowsOp(head.getOperation());

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

/// Whether any two-qubit or single-qubit unitary op (including `ctrl` shells)
/// remains off the native gateset. Used as the pass' final convergence check.
static bool hasNonNativeOps(Operation* root, const NativeGateset& spec) {
  const WalkResult walkResult = root->walk([&](Operation* op) {
    if (!isWalkableUnitaryShell(op) || !llvm::isa<UnitaryOpInterface>(op)) {
      return WalkResult::advance();
    }
    return spec.allowsOp(op) ? WalkResult::advance() : WalkResult::interrupt();
  });
  return walkResult.wasInterrupted();
}

namespace {

/// Fuses a maximal window of two-qubit gates (plus interleaved single-qubit
/// gates) into a single composed unitary and resynthesizes it to the native
/// gateset when that is beneficial.
struct FuseTwoQubitWindowPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  FuseTwoQubitWindowPattern(MLIRContext* ctx, NativeGateset specIn)
      : OpInterfaceRewritePattern(ctx), spec(std::move(specIn)) {}

  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    // Only anchor on a run start: a run member not fed by an earlier window.
    if (!isTwoQubitRunMember(op) || feedsFromSameTwoQubitWindow(op)) {
      return failure();
    }

    FusableTwoQubitRun run = scanFusableTwoQubitRun(op, spec);
    if (run.ops.size() < 2) {
      return failure();
    }

    // Replace when off-gateset ops must be lowered, or when resynthesis uses
    // fewer entanglers than the fused window.
    const auto native = spec.decomposeTarget(run.composed);
    if (!native || (!run.anyNonNative && native->numBasisUses >= run.numTwoQ)) {
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
    for (Operation* member : llvm::reverse(run.ops)) {
      rewriter.eraseOp(member);
    }
    return success();
  }

  NativeGateset spec;
};

/// Lowers a single off-gateset two-qubit op (bare or single-target `CtrlOp`)
/// to the native entangler plus native single-qubit factors via Weyl synthesis.
/// Native two-qubit ops and windows fusable by @ref FuseTwoQubitWindowPattern
/// are left to that pattern.
struct LowerTwoQubitOpPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  LowerTwoQubitOpPattern(MLIRContext* ctx, NativeGateset specIn)
      : OpInterfaceRewritePattern(ctx), spec(std::move(specIn)) {}

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

  NativeGateset spec;
};

} // namespace

/// Fuses single-qubit runs (and lowers lone off-gateset single-qubit ops) by
/// reusing the `fuse-single-qubit-unitary-runs` rewrite.
///
/// `qco.ctrl` bodies are skipped so the `X`/`Z` bodies of native entanglers are
/// preserved (both before and after two-qubit synthesis).
static LogicalResult fuseSingleQubitRuns(ModuleOp module,
                                         const EulerBasis basis) {
  RewritePatternSet patterns(module.getContext());
  populateFuseSingleQubitUnitaryRunsPatterns(patterns, basis,
                                             /*skipControlledBodies=*/true);
  return applyPatternsGreedily(module, std::move(patterns));
}

/// Fuses two-qubit windows, then lowers any remaining off-gateset two-qubit
/// ops.
static LogicalResult fuseAndLowerTwoQubitOps(ModuleOp module,
                                             const NativeGateset& spec) {
  MLIRContext* ctx = module.getContext();
  {
    RewritePatternSet windowPatterns(ctx);
    windowPatterns.add<FuseTwoQubitWindowPattern>(ctx, spec);
    if (failed(applyPatternsGreedily(module, std::move(windowPatterns)))) {
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
    // 2. Fuse two-qubit windows and lower remaining off-gateset two-qubit ops.
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
