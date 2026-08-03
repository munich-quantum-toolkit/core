/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/SynthesisBasis.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/Utils/Transforms/GlobalPhaseNormalization.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h> // IWYU pragma: keep (Passes.h.inc)
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

namespace mlir::qco {

using decomposition::NativeSynthesisBasis;
using decomposition::synthesizeUnitary2QWeyl;

namespace {

/** Composed unitary and metadata for a fusable two-qubit run. */
struct FusableTwoQubitRun {
  SmallVector<Operation*, 8> ops; ///< Members in program order.
  Matrix4x4 composed = Matrix4x4::identity();
  unsigned numTwoQ = 0; ///< Number of two-qubit members (entanglers consumed).
  Value tailA;          ///< Current output wires of the run's tail.
  Value tailB;
};

} // namespace

// --- Run membership ------------------------------------------------------- //

/// Whether `op` is nested under a modifier body. Such unitaries are handled
/// through their shell op, so the top-level walk skips them.
static bool isExcludedFromTopLevelUnitaryWalk(Operation* op) {
  return op->getParentOfType<CtrlOp>() || op->getParentOfType<InvOp>() ||
         op->getParentOfType<PowOp>();
}

/// Whether `op` is a unitary shell the pass may rewrite at top level.
static bool isWalkableUnitaryShell(Operation* op) {
  return !isa<BarrierOp, GPhaseOp>(op) &&
         !isExcludedFromTopLevelUnitaryWalk(op);
}

/// Builds the constant 4x4 matrix for a two-qubit op (bare or single-target
/// `CtrlOp`). Returns false for a `CtrlOp` that is not
/// single-control/single-target, or an op whose matrix is not known at compile
/// time.
static bool assignTwoQubitOpMatrix(Operation* op, Matrix4x4& matrix) {
  if (auto ctrl = dyn_cast<CtrlOp>(op)) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    return cast<UnitaryOpInterface>(ctrl.getOperation())
        .getUnitaryMatrix4x4(matrix);
  }
  auto unitary = cast<UnitaryOpInterface>(op);
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
  auto unitary = dyn_cast<UnitaryOpInterface>(*wire.user_begin());
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
    auto unitary = dyn_cast<UnitaryOpInterface>(def);
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
                                  UnitaryOpInterface op) {
  Matrix4x4 opMatrix;
  [[maybe_unused]] const bool assigned =
      assignTwoQubitOpMatrix(op.getOperation(), opMatrix);
  assert(assigned && "a two-qubit run member always exposes a 4x4 matrix");
  const Value in0 = op.getInputQubit(0);
  const Value in1 = op.getInputQubit(1);
  size_t id0 = 0;
  size_t id1 = 1;
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
}

/// Appends a single-qubit gate on run wire `wireIndex` (0 = A, 1 = B).
static void absorbOneQubitIntoRun(FusableTwoQubitRun& run,
                                  UnitaryOpInterface op, unsigned wireIndex) {
  Matrix2x2 raw;
  [[maybe_unused]] const bool assigned = op.getUnitaryMatrix2x2(raw);
  assert(assigned && "a single-qubit run member always exposes a 2x2 matrix");
  run.composed.premultiplyBy(raw.embedInTwoQubit(wireIndex));
  run.ops.push_back(op.getOperation());
  (wireIndex == 0 ? run.tailA : run.tailB) = op.getOutputQubit(0);
}

/// Walks forward from `head`, composing the run's matrix and metadata. Absorbs
/// a following two-qubit gate when it keeps both run wires together, otherwise
/// the single-qubit gate first in program order; stops at the first boundary
/// that would split the run's two wires.
static FusableTwoQubitRun scanFusableTwoQubitRun(UnitaryOpInterface head) {
  FusableTwoQubitRun run;
  [[maybe_unused]] const bool assigned =
      assignTwoQubitOpMatrix(head.getOperation(), run.composed);
  assert(assigned && "a run head is a two-qubit member with a 4x4 matrix");
  run.tailA = head.getOutputQubit(0);
  run.tailB = head.getOutputQubit(1);
  run.ops.push_back(head.getOperation());
  run.numTwoQ = 1;

  while (true) {
    UnitaryOpInterface nextOnA = uniqueUnitaryUser(run.tailA);
    UnitaryOpInterface nextOnB = uniqueUnitaryUser(run.tailB);
    const bool sameOp =
        nextOnA && nextOnB && nextOnA.getOperation() == nextOnB.getOperation();

    if (sameOp && nextOnA.isTwoQubit()) {
      absorbTwoQubitIntoRun(run, nextOnA);
      continue;
    }

    const bool aSingle = nextOnA && nextOnA.isSingleQubit() && !sameOp;
    const bool bSingle = nextOnB && nextOnB.isSingleQubit() && !sameOp;
    if (aSingle && bSingle && nextOnA->getBlock() != nextOnB->getBlock()) {
      break;
    }
    if (aSingle && (!bSingle || nextOnA->isBeforeInBlock(nextOnB))) {
      absorbOneQubitIntoRun(run, nextOnA, /*wireIndex=*/0);
      continue;
    }
    if (bSingle) {
      absorbOneQubitIntoRun(run, nextOnB, /*wireIndex=*/1);
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

namespace {

/// Fuses a maximal constant run only when generic resynthesis strictly reduces
/// its two-qubit operation count.
struct OptimizeTwoQubitUnitaryRunsPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  OptimizeTwoQubitUnitaryRunsPattern(MLIRContext* ctx,
                                     NativeSynthesisBasis basisIn,
                                     bool& changedIn)
      : OpInterfaceRewritePattern(ctx), basis(basisIn), changed(&changedIn) {}

  NativeSynthesisBasis basis;
  bool* changed;

  /// Whether `op` anchors a run: a two-qubit run member whose two wires are not
  /// both fed by the same earlier run (which would make it a continuation).
  static bool isRunStart(UnitaryOpInterface op) {
    return isTwoQubitRunMember(op) && !feedsFromSameTwoQubitRun(op);
  }

  /// Fuse the run anchored at `op` only if it has multiple constant members and
  /// Weyl resynthesis uses fewer entanglers than the original run.
  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    if (!isRunStart(op)) {
      return failure();
    }

    FusableTwoQubitRun run = scanFusableTwoQubitRun(op);
    if (run.ops.size() < 2) {
      return failure();
    }

    const auto native = basis.decomposeTarget(run.composed);
    if (native.numBasisUses >= run.numTwoQ) {
      return failure();
    }

    auto firstOp = cast<UnitaryOpInterface>(run.ops.front());
    rewriter.setInsertionPoint(firstOp);
    const auto synthesized = synthesizeUnitary2QWeyl(
        rewriter, firstOp.getLoc(), firstOp.getInputQubit(0),
        firstOp.getInputQubit(1), run.composed, basis);
    if (failed(synthesized)) {
      firstOp->emitError("failed to emit synthesized two-qubit gate sequence");
      return failure();
    }
    decomposition::emitGPhaseIfNeeded(rewriter, firstOp.getLoc(),
                                      synthesized->globalPhase);
    rewriter.replaceAllUsesWith(run.tailA, synthesized->qubit0);
    rewriter.replaceAllUsesWith(run.tailB, synthesized->qubit1);
    eraseFusableRun(rewriter, run);
    *changed = true;
    return success();
  }
};

} // namespace

using SiteId = CompilerTarget::SiteId;

static FailureOr<SiteId> resolveProviderSite(Value value) {
  llvm::SmallDenseSet<Value, 16> visited;
  while (value) {
    if (!visited.insert(value).second) {
      return failure();
    }

    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Operation* const parent = argument.getOwner()->getParentOp();
      const auto index = argument.getArgNumber();
      if (auto ifOp = dyn_cast_or_null<IfOp>(parent);
          ifOp && index < ifOp.getQubits().size()) {
        value = ifOp.getQubits()[index];
        continue;
      }
      if (auto switchOp = dyn_cast_or_null<IndexSwitchOp>(parent);
          switchOp && index < switchOp.getTargets().size()) {
        value = switchOp.getTargets()[index];
        continue;
      }
      if (auto forOp = dyn_cast_or_null<scf::ForOp>(parent);
          forOp && index > 0 && index <= forOp.getInits().size()) {
        value = forOp.getInits()[index - 1];
        continue;
      }
      if (auto whileOp = dyn_cast_or_null<scf::WhileOp>(parent);
          whileOp && index < whileOp.getInits().size()) {
        value = whileOp.getInits()[index];
        continue;
      }
      return failure();
    }

    Operation* const definingOp = value.getDefiningOp();
    if (auto staticOp = dyn_cast_or_null<StaticOp>(definingOp)) {
      return static_cast<SiteId>(staticOp.getIndex());
    }
    if (auto unitary = dyn_cast_or_null<UnitaryOpInterface>(definingOp)) {
      value = unitary.getInputForOutput(value);
      continue;
    }
    if (auto measureOp = dyn_cast_or_null<MeasureOp>(definingOp)) {
      value = measureOp.getQubitIn();
      continue;
    }
    if (auto resetOp = dyn_cast_or_null<ResetOp>(definingOp)) {
      value = resetOp.getQubitIn();
      continue;
    }
    if (auto forOp = dyn_cast_or_null<scf::ForOp>(definingOp)) {
      auto result = dyn_cast<OpResult>(value);
      if (!result) {
        return failure();
      }
      value = forOp.getTiedLoopInit(result)->get();
      continue;
    }
    if (auto whileOp = dyn_cast_or_null<scf::WhileOp>(definingOp)) {
      auto result = dyn_cast<OpResult>(value);
      if (!result || result.getResultNumber() >= whileOp.getInits().size()) {
        return failure();
      }
      value = whileOp.getInits()[result.getResultNumber()];
      continue;
    }
    if (auto ifOp = dyn_cast_or_null<IfOp>(definingOp)) {
      auto result = dyn_cast<OpResult>(value);
      if (!result) {
        return failure();
      }
      OpOperand* const input = ifOp.getTiedQubit(result);
      if (input == nullptr) {
        return failure();
      }
      value = input->get();
      continue;
    }
    if (auto switchOp = dyn_cast_or_null<IndexSwitchOp>(definingOp)) {
      auto result = dyn_cast<OpResult>(value);
      if (!result) {
        return failure();
      }
      OpOperand* const input = switchOp.getTiedTarget(result);
      if (input == nullptr) {
        return failure();
      }
      value = input->get();
      continue;
    }
    return failure();
  }
  return failure();
}

static FailureOr<SmallVector<SiteId>> providerLocus(Operation* operation) {
  SmallVector<Value> qubits;
  if (auto unitary = dyn_cast<UnitaryOpInterface>(operation)) {
    llvm::append_range(qubits, unitary.getInputQubits());
  } else if (auto measureOp = dyn_cast<MeasureOp>(operation)) {
    qubits.emplace_back(measureOp.getQubitIn());
  } else if (auto resetOp = dyn_cast<ResetOp>(operation)) {
    qubits.emplace_back(resetOp.getQubitIn());
  } else {
    return failure();
  }

  SmallVector<SiteId> locus;
  locus.reserve(qubits.size());
  for (const Value qubit : qubits) {
    const auto site = resolveProviderSite(qubit);
    if (failed(site)) {
      return failure();
    }
    locus.emplace_back(*site);
  }
  return locus;
}

static bool requiresTargetSynthesis(Operation* operation,
                                    const CompilerTarget& target,
                                    const ArrayRef<SiteId> locus) {
  return !target.supports(operation, locus);
}

/// Whether a target-selected entangler must use the reverse provider locus.
///
/// `CompilerTarget` selects an operand-symmetric gate as globally usable when
/// either ordered orientation is available. Preserve the logical wire order
/// while emitting that gate at the provider-supported orientation.
static bool useReverseEntanglerLocus(const CompilerTarget& target,
                                     const CompilerTarget::GateKind entangler,
                                     const ArrayRef<SiteId> locus) {
  assert(locus.size() == 2 && "a synthesis entangler has two provider sites");
  if (target.supports(entangler, locus)) {
    return false;
  }
  const std::array reverseLocus{locus[1], locus[0]};
  assert(target.supports(entangler, reverseLocus) &&
         "a globally selected entangler is available in one orientation");
  return true;
}

static void appendProviderLocus(InFlightDiagnostic& diagnostic,
                                const ArrayRef<SiteId> locus) {
  diagnostic << '[';
  for (const auto [index, site] : llvm::enumerate(locus)) {
    if (index != 0) {
      diagnostic << ", ";
    }
    diagnostic << site;
  }
  diagnostic << ']';
}

namespace {

struct SynthesisNeed {
  Operation* operation;
  SmallVector<SiteId> locus;
};

} // namespace

static std::optional<SynthesisNeed>
findSynthesisNeed(Operation* root, const CompilerTarget& target) {
  std::optional<SynthesisNeed> need;
  root->walk([&](Operation* operation) {
    auto unitary = dyn_cast<UnitaryOpInterface>(operation);
    if (!unitary || !isWalkableUnitaryShell(operation) ||
        (unitary.getNumQubits() != 1 && unitary.getNumQubits() != 2)) {
      return WalkResult::advance();
    }
    auto locus = providerLocus(operation);
    if (failed(locus) || !requiresTargetSynthesis(operation, target, *locus)) {
      return WalkResult::advance();
    }
    need.emplace(
        SynthesisNeed{.operation = operation, .locus = std::move(*locus)});
    return WalkResult::interrupt();
  });
  return need;
}

namespace {

struct LowerTargetSingleQubitOpPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  LowerTargetSingleQubitOpPattern(MLIRContext* context,
                                  const CompilerTarget& targetIn,
                                  NativeSynthesisBasis basisIn)
      : OpInterfaceRewritePattern(context), target(targetIn), basis(basisIn) {}

  CompilerTarget target;
  NativeSynthesisBasis basis;

  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    Operation* const operation = op.getOperation();
    if (!op.isSingleQubit() || !isWalkableUnitaryShell(operation)) {
      return failure();
    }
    const auto locus = providerLocus(operation);
    if (failed(locus) || !requiresTargetSynthesis(operation, target, *locus)) {
      return failure();
    }
    Matrix2x2 matrix;
    if (!op.getUnitaryMatrix2x2(matrix)) {
      return failure();
    }

    rewriter.setInsertionPoint(operation);
    const auto synthesized = decomposition::synthesizeUnitary1QEuler(
        rewriter, operation->getLoc(), op.getInputQubit(0), matrix,
        /*runSize=*/1, /*hasNonBasisGate=*/true, basis.singleQubit);
    if (!synthesized) {
      return failure();
    }
    decomposition::emitGPhaseIfNeeded(rewriter, operation->getLoc(),
                                      synthesized->globalPhase);
    rewriter.replaceOp(operation, synthesized->qubit);
    return success();
  }
};

struct LowerTargetTwoQubitOpPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  LowerTargetTwoQubitOpPattern(MLIRContext* context,
                               const CompilerTarget& targetIn,
                               NativeSynthesisBasis basisIn)
      : OpInterfaceRewritePattern(context), target(targetIn), basis(basisIn) {}

  CompilerTarget target;
  NativeSynthesisBasis basis;

  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    Operation* const operation = op.getOperation();
    if (!op.isTwoQubit() || !isWalkableUnitaryShell(operation)) {
      return failure();
    }
    const auto locus = providerLocus(operation);
    if (failed(locus) || !requiresTargetSynthesis(operation, target, *locus)) {
      return failure();
    }
    Matrix4x4 matrix;
    if (!assignTwoQubitOpMatrix(operation, matrix)) {
      return failure();
    }

    Value input0;
    Value input1;
    if (auto ctrl = dyn_cast<CtrlOp>(operation)) {
      input0 = ctrl.getInputControl(0);
      input1 = ctrl.getInputTarget(0);
    } else {
      input0 = op.getInputQubit(0);
      input1 = op.getInputQubit(1);
    }

    rewriter.setInsertionPoint(operation);
    const auto synthesized = synthesizeUnitary2QWeyl(
        rewriter, operation->getLoc(), input0, input1, matrix, basis,
        useReverseEntanglerLocus(target, basis.entangler, *locus));
    if (failed(synthesized)) {
      return failure();
    }
    decomposition::emitGPhaseIfNeeded(rewriter, operation->getLoc(),
                                      synthesized->globalPhase);
    rewriter.replaceOp(operation,
                       ValueRange{synthesized->qubit0, synthesized->qubit1});
    return success();
  }
};

} // namespace

static LogicalResult optimizeTwoQubitRuns(ModuleOp moduleOp) {
  const auto basis =
      NativeSynthesisBasis::fromCompilerTarget(CompilerTarget::SynthesisBasis{
          .singleQubit = CompilerTarget::SingleQubitBasis::U,
          .entangler = CompilerTarget::GateKind::CX});
  bool changed = false;
  RewritePatternSet patterns(moduleOp.getContext());
  patterns.add<OptimizeTwoQubitUnitaryRunsPattern>(moduleOp.getContext(), basis,
                                                   changed);
  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    return failure();
  }
  if (!changed) {
    return success();
  }
  return mlir::mqt::normalizeGlobalPhases(moduleOp);
}

namespace {

struct OptimizeTwoQubitUnitaryRunsPass final
    : PassWrapper<OptimizeTwoQubitUnitaryRunsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeTwoQubitUnitaryRunsPass)

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<QCODialect, arith::ArithDialect>();
  }

protected:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    if (failed(optimizeTwoQubitRuns(moduleOp))) {
      signalPassFailure();
    }
  }
};

struct TargetNativeSynthesisPass final
    : PassWrapper<TargetNativeSynthesisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TargetNativeSynthesisPass)

  explicit TargetNativeSynthesisPass(const CompilerTarget& targetIn)
      : target(targetIn) {}

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<QCODialect, arith::ArithDialect>();
  }

protected:
  void runOnOperation() override {
    if (!target.hasExplicitOperations()) {
      return;
    }
    ModuleOp moduleOp = getOperation();
    auto need = findSynthesisNeed(moduleOp, target);
    if (!need) {
      return;
    }

    const auto targetBasis = target.synthesisBasis();
    if (!targetBasis) {
      auto diagnostic = need->operation->emitError()
                        << "target-native synthesis cannot lower operation '"
                        << need->operation->getName()
                        << "' at ordered provider locus ";
      appendProviderLocus(diagnostic, need->locus);
      diagnostic << ": the target has no globally usable synthesis basis";
      signalPassFailure();
      return;
    }
    const auto basis = NativeSynthesisBasis::fromCompilerTarget(*targetBasis);
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerTargetSingleQubitOpPattern, LowerTargetTwoQubitOpPattern>(
        &getContext(), target, basis);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    need = findSynthesisNeed(moduleOp, target);
    if (need) {
      auto diagnostic = need->operation->emitError()
                        << "target-native synthesis cannot lower operation '"
                        << need->operation->getName()
                        << "' at ordered provider locus ";
      appendProviderLocus(diagnostic, need->locus);
      diagnostic << ": its unitary matrix is not available at compile time";
      signalPassFailure();
      return;
    }
    if (failed(mlir::mqt::normalizeGlobalPhases(moduleOp))) {
      signalPassFailure();
    }
  }

  CompilerTarget target;
};

struct VerifyTargetConformancePass final
    : PassWrapper<VerifyTargetConformancePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyTargetConformancePass)

  explicit VerifyTargetConformancePass(const CompilerTarget& targetIn)
      : target(targetIn) {}

protected:
  void runOnOperation() override {
    WalkResult result = getOperation()->walk([&](Operation* operation) {
      size_t parameterCount = 0;
      if (auto unitary = dyn_cast<UnitaryOpInterface>(operation)) {
        if (isExcludedFromTopLevelUnitaryWalk(operation)) {
          return WalkResult::advance();
        }
        parameterCount = unitary.getNumParams();
      } else if (!isa<MeasureOp, ResetOp>(operation)) {
        return WalkResult::advance();
      }

      const auto locus = providerLocus(operation);
      if (failed(locus)) {
        operation->emitError()
            << "target conformance requires every hardware qubit operand to "
               "trace to a qco.static provider site";
        return WalkResult::interrupt();
      }
      if (target.supports(operation, *locus)) {
        return WalkResult::advance();
      }

      auto diagnostic = operation->emitError()
                        << "target does not support operation '"
                        << operation->getName() << "' with arity "
                        << locus->size() << " and " << parameterCount
                        << " parameter(s) at ordered provider locus ";
      appendProviderLocus(diagnostic, *locus);
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }

  CompilerTarget target;
};

} // namespace

std::unique_ptr<Pass> createOptimizeTwoQubitUnitaryRuns() {
  return std::make_unique<OptimizeTwoQubitUnitaryRunsPass>();
}

std::unique_ptr<Pass>
createTargetNativeSynthesis(const CompilerTarget& target) {
  return std::make_unique<TargetNativeSynthesisPass>(target);
}

std::unique_ptr<Pass>
createVerifyTargetConformance(const CompilerTarget& target) {
  return std::make_unique<VerifyTargetConformancePass>(target);
}

} // namespace mlir::qco
