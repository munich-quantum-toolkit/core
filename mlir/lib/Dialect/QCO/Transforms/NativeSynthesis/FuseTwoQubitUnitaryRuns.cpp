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
#include "mlir/Dialect/QCO/Transforms/Decomposition/NativeProfile.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_FUSETWOQUBITUNITARYRUNS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

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

static void collectUnitaryOpsInPreOrder(Operation* root,
                                        SmallVectorImpl<Operation*>& ops) {
  root->walk([&](Operation* op) {
    if (isExcludedFromTopLevelUnitaryWalk(op)) {
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
  return *decomposition::synthesizeUnitary1QEuler(
      rewriter, loc, inQubit, matrix, /*runSize=*/0,
      /*hasNonBasisGate=*/true, basis);
}

static bool assignTwoQubitOpMatrix(Operation* op, Matrix4x4& matrix) {
  if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
    return false;
  }
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    Operation* body = ctrl.getBodyUnitary(0).getOperation();
    if (llvm::isa<XOp>(body)) {
      matrix = twoQubitControlledX01();
      return true;
    }
    if (llvm::isa<ZOp>(body)) {
      matrix = twoQubitControlledZ();
      return true;
    }
    return false;
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

static UnitaryOpInterface fusibleSingleQubitOp(Operation* op) {
  if (llvm::isa<CtrlOp>(op)) {
    return {};
  }
  auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
  return isOneQubitWindowMember(unitary) ? unitary : UnitaryOpInterface{};
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

static bool isTwoQubitRunStart(UnitaryOpInterface op) {
  return isTwoQubitRunMember(op) && !feedsFromSameTwoQubitWindow(op);
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

struct OneQubitRun {
  SmallVector<UnitaryOpInterface, 4> ops;
};

} // namespace

static void
markNonNativeIfNeeded(FusableTwoQubitRun& run, Operation* op,
                      const decomposition::NativeProfileSpec& spec) {
  if (!decomposition::allowsOp(op, spec)) {
    run.anyNonNative = true;
  }
}

// Replace when off-menu ops must be lowered, or when resynthesis uses fewer
// entanglers than the fused window.
static bool shouldApplyTwoQubitRunReplacement(const FusableTwoQubitRun& run,
                                              std::uint8_t numBasisUses) {
  if (run.anyNonNative) {
    return true;
  }
  return numBasisUses < run.numTwoQ;
}

static void
absorbTwoQubitIntoRun(FusableTwoQubitRun& run, UnitaryOpInterface op,
                      const decomposition::NativeProfileSpec& spec) {
  Matrix4x4 opMatrix;
  if (!assignTwoQubitOpMatrix(op.getOperation(), opMatrix)) {
    return;
  }
  const Value in0 = op.getInputQubit(0);
  const Value in1 = op.getInputQubit(1);
  SmallVector<std::size_t, 2> ids;
  if (in0 == run.tailA && in1 == run.tailB) {
    ids = {0, 1};
    run.tailA = op.getOutputQubit(0);
    run.tailB = op.getOutputQubit(1);
  } else if (in0 == run.tailB && in1 == run.tailA) {
    ids = {1, 0};
    run.tailA = op.getOutputQubit(1);
    run.tailB = op.getOutputQubit(0);
  } else {
    return;
  }
  run.composed = opMatrix.reorderForQubits(ids[0], ids[1]) * run.composed;
  run.ops.push_back(op.getOperation());
  ++run.numTwoQ;
  markNonNativeIfNeeded(run, op.getOperation(), spec);
}

static void absorbOneQubitIntoRun(FusableTwoQubitRun& run,
                                  UnitaryOpInterface op,
                                  const decomposition::NativeProfileSpec& spec,
                                  unsigned wireIndex) {
  Matrix2x2 raw;
  if (!op.getUnitaryMatrix2x2(raw)) {
    return;
  }
  const auto pad = raw.embedInTwoQubit(wireIndex);
  run.composed = pad * run.composed;
  run.ops.push_back(op.getOperation());
  markNonNativeIfNeeded(run, op.getOperation(), spec);
  if (wireIndex == 0) {
    run.tailA = op.getOutputQubit(0);
  } else {
    run.tailB = op.getOutputQubit(0);
  }
}

static FusableTwoQubitRun
scanFusableTwoQubitRun(UnitaryOpInterface head,
                       const decomposition::NativeProfileSpec& spec) {
  FusableTwoQubitRun run;
  run.tailA = head.getOutputQubit(0);
  run.tailB = head.getOutputQubit(1);
  run.ops.push_back(head.getOperation());
  run.numTwoQ = 1;
  if (!assignTwoQubitOpMatrix(head.getOperation(), run.composed)) {
    run.composed = Matrix4x4::identity();
    run.numTwoQ = 0;
    run.ops.clear();
    return run;
  }
  markNonNativeIfNeeded(run, head.getOperation(), spec);

  while (true) {
    UnitaryOpInterface nextOnA = uniqueUnitaryUser(run.tailA);
    UnitaryOpInterface nextOnB = uniqueUnitaryUser(run.tailB);

    if (nextOnA && nextOnB &&
        nextOnA.getOperation() == nextOnB.getOperation() &&
        nextOnA.isTwoQubit()) {
      absorbTwoQubitIntoRun(run, nextOnA, spec);
      continue;
    }

    if (nextOnA && nextOnB &&
        nextOnA.getOperation() != nextOnB.getOperation() &&
        nextOnA.isSingleQubit() && nextOnB.isSingleQubit()) {
      if (nextOnA->isBeforeInBlock(nextOnB)) {
        absorbOneQubitIntoRun(run, nextOnA, spec, /*wireIndex=*/0);
        continue;
      }
      absorbOneQubitIntoRun(run, nextOnB, spec, /*wireIndex=*/1);
      continue;
    }

    if (nextOnA && nextOnA.isSingleQubit() &&
        (!nextOnB || nextOnA.getOperation() != nextOnB.getOperation())) {
      absorbOneQubitIntoRun(run, nextOnA, spec, /*wireIndex=*/0);
      continue;
    }

    if (nextOnB && nextOnB.isSingleQubit() &&
        (!nextOnA || nextOnB.getOperation() != nextOnA.getOperation())) {
      absorbOneQubitIntoRun(run, nextOnB, spec, /*wireIndex=*/1);
      continue;
    }

    break;
  }
  return run;
}

static void eraseFusableTwoQubitRun(PatternRewriter& rewriter,
                                    const FusableTwoQubitRun& run) {
  for (Operation* op : llvm::reverse(run.ops)) {
    rewriter.eraseOp(op);
  }
}

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
    return !decomposition::allowsOp(u.getOperation(), spec);
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

static bool hasNonNativeOps(Operation* root,
                            const decomposition::NativeProfileSpec& spec,
                            bool singleQubitOnly) {
  const mlir::WalkResult walkResult = root->walk([&](Operation* op) {
    if (!isWalkableUnitaryShell(op)) {
      return mlir::WalkResult::advance();
    }
    if (singleQubitOnly) {
      auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
      if (!unitary || !unitary.isSingleQubit()) {
        return mlir::WalkResult::advance();
      }
    } else if (!llvm::isa<CtrlOp>(op) && !llvm::isa<UnitaryOpInterface>(op)) {
      return mlir::WalkResult::advance();
    }
    if (!decomposition::allowsOp(op, spec)) {
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return walkResult.wasInterrupted();
}

static LogicalResult synthesizeTwoQubitOp(
    IRRewriter& rewriter, Operation* op, Location loc, Value in0, Value in1,
    const decomposition::NativeProfileSpec& spec, llvm::Twine matrixErrorMsg,
    llvm::Twine synthesisErrorMsg) {
  if (decomposition::allowsOp(op, spec)) {
    return success();
  }
  Matrix4x4 matrix;
  if (!assignTwoQubitOpMatrix(op, matrix)) {
    op->emitError(matrixErrorMsg);
    return failure();
  }
  rewriter.setInsertionPoint(op);
  Value out0;
  Value out1;
  if (failed(decomposition::synthesizeUnitary2QWeyl(
          rewriter, loc, in0, in1, matrix, spec, out0, out1))) {
    op->emitError(synthesisErrorMsg);
    return failure();
  }
  rewriter.replaceOp(op, ValueRange{out0, out1});
  return success();
}

namespace {

struct FuseTwoQubitWindowPattern
    : public OpInterfaceRewritePattern<UnitaryOpInterface> {
  FuseTwoQubitWindowPattern(MLIRContext* ctx,
                            decomposition::NativeProfileSpec specIn)
      : OpInterfaceRewritePattern(ctx), spec(std::move(specIn)) {}

  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    if (!isTwoQubitRunStart(op)) {
      return failure();
    }

    FusableTwoQubitRun run = scanFusableTwoQubitRun(op, spec);
    if (run.ops.size() < 2) {
      return failure();
    }

    const auto numBasisUses =
        decomposition::twoQubitEntanglerCount(run.composed, spec);
    if (!numBasisUses ||
        !shouldApplyTwoQubitRunReplacement(run, *numBasisUses)) {
      return failure();
    }

    Operation* firstOp = run.ops.front();
    auto firstUnitary = llvm::cast<UnitaryOpInterface>(firstOp);
    const Value inA = firstUnitary.getInputQubit(0);
    const Value inB = firstUnitary.getInputQubit(1);

    rewriter.setInsertionPoint(firstOp);
    Value newA;
    Value newB;
    if (failed(decomposition::synthesizeUnitary2QWeyl(
            rewriter, firstOp->getLoc(), inA, inB, run.composed, spec, newA,
            newB))) {
      firstOp->emitError("failed to emit synthesized two-qubit gate sequence");
      return failure();
    }
    rewriter.replaceAllUsesWith(run.tailA, newA);
    rewriter.replaceAllUsesWith(run.tailB, newB);
    eraseFusableTwoQubitRun(rewriter, run);
    return success();
  }

  decomposition::NativeProfileSpec spec;
};

} // namespace

static LogicalResult
fuseTwoQubitUnitaryRuns(Operation* root,
                        const decomposition::NativeProfileSpec& spec) {
  RewritePatternSet patterns(root->getContext());
  patterns.add<FuseTwoQubitWindowPattern>(patterns.getContext(), spec);
  return applyPatternsGreedily(root, std::move(patterns));
}

namespace {

struct FuseTwoQubitUnitaryRunsPass
    : impl::FuseTwoQubitUnitaryRunsBase<FuseTwoQubitUnitaryRunsPass> {
  FuseTwoQubitUnitaryRunsPass() = default;

  explicit FuseTwoQubitUnitaryRunsPass(FuseTwoQubitUnitaryRunsOptions options)
      : FuseTwoQubitUnitaryRunsBase(std::move(options)) {}

protected:
  void runOnOperation() override {
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
    const decomposition::EulerBasis oneQubitBasis = spec.eulerBasis();

    IRRewriter rewriter(&getContext());

    fuseOneQubitRuns(rewriter, spec, oneQubitBasis);
    if (failed(fuseTwoQubitUnitaryRuns(getOperation(), spec))) {
      signalPassFailure();
      return;
    }
    constexpr unsigned kMaxSynthesisSweeps = 4;
    for (unsigned i = 0; i < kMaxSynthesisSweeps; ++i) {
      if (failed(synthesizeRemainingOps(rewriter, spec, oneQubitBasis))) {
        signalPassFailure();
        return;
      }
      if (!hasNonNativeOps(getOperation(), spec, /*singleQubitOnly=*/true)) {
        break;
      }
    }
    if (hasNonNativeOps(getOperation(), spec, /*singleQubitOnly=*/true)) {
      getOperation().emitError()
          << "native gate synthesis did not converge within "
          << kMaxSynthesisSweeps
          << " sweeps (single-qubit ops remain outside the native menu)";
      signalPassFailure();
      return;
    }
    fuseOneQubitRuns(rewriter, spec, oneQubitBasis);
    constexpr unsigned kPostMenuCleanupSweeps = 4;
    unsigned postMenuSweepsRemaining = kPostMenuCleanupSweeps;
    while (hasNonNativeOps(getOperation(), spec, /*singleQubitOnly=*/false) &&
           postMenuSweepsRemaining-- > 0) {
      if (failed(synthesizeRemainingOps(rewriter, spec, oneQubitBasis))) {
        signalPassFailure();
        return;
      }
      fuseOneQubitRuns(rewriter, spec, oneQubitBasis);
    }
    if (hasNonNativeOps(getOperation(), spec, /*singleQubitOnly=*/false)) {
      getOperation().emitError()
          << "native gate synthesis: operations remain outside the native menu "
             "after final cleanup";
      signalPassFailure();
      return;
    }
  }

private:
  void fuseOneQubitRuns(IRRewriter& rewriter,
                        const decomposition::NativeProfileSpec& spec,
                        const decomposition::EulerBasis basis) {
    SmallVector<OneQubitRun> runs;
    llvm::DenseMap<Operation*, size_t> tailOpToRun;

    // Require single-use tail output so fan-out wires are not fused away.
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

  LogicalResult
  synthesizeRemainingOps(IRRewriter& rewriter,
                         const decomposition::NativeProfileSpec& spec,
                         const decomposition::EulerBasis basis) {
    SmallVector<Operation*, 32> ops;
    collectUnitaryOpsInPreOrder(getOperation(), ops);
    llvm::DenseSet<Operation*> erasedOps;

    for (Operation* op : ops) {
      if (erasedOps.contains(op)) {
        continue;
      }
      if (!isWalkableUnitaryShell(op)) {
        continue;
      }

      if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
        const bool wasAlreadyNative = decomposition::allowsOp(op, spec);
        if (failed(synthesizeControlled(rewriter, ctrl, spec))) {
          return failure();
        }
        if (!wasAlreadyNative) {
          erasedOps.insert(op);
        }
        continue;
      }

      auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
      if (!unitary) {
        continue;
      }

      if (unitary.isSingleQubit()) {
        if (!decomposition::allowsOp(op, spec)) {
          if (failed(rewriteSingleQubit(rewriter, op, unitary, basis))) {
            return failure();
          }
          erasedOps.insert(op);
        }
        continue;
      }

      if (unitary.isTwoQubit()) {
        if (failed(synthesizeBareTwoQubit(rewriter, op, unitary, spec))) {
          return failure();
        }
        erasedOps.insert(op);
      }
    }
    return success();
  }

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

  static LogicalResult
  synthesizeControlled(IRRewriter& rewriter, CtrlOp ctrl,
                       const decomposition::NativeProfileSpec& spec) {
    return synthesizeTwoQubitOp(
        rewriter, ctrl.getOperation(), ctrl.getLoc(), ctrl.getInputControl(0),
        ctrl.getInputTarget(0), spec,
        "native synthesis: cannot build a constant 4x4 matrix for this "
        "controlled gate (unsupported body or non-constant parameters)",
        "controlled gate not allowed by selected profile");
  }

  static LogicalResult
  synthesizeBareTwoQubit(IRRewriter& rewriter, Operation* op,
                         UnitaryOpInterface unitary,
                         const decomposition::NativeProfileSpec& spec) {
    return synthesizeTwoQubitOp(
        rewriter, op, op->getLoc(), unitary.getInputQubit(0),
        unitary.getInputQubit(1), spec,
        "unsupported two-qubit operation for selected profile",
        "unsupported two-qubit operation for selected profile");
  }
};

} // namespace

} // namespace mlir::qco
