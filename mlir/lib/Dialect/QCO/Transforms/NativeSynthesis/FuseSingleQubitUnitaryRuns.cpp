/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QCO/Utils/Matrix.h"
#include "mqt/Dialect/QCO/Utils/WireIterator.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h" // IWYU pragma: keep (Passes.h.inc)

#include "mlir/Dialect/Arith/IR/Arith.h" // IWYU pragma: keep (Passes.h.inc)
#include "mlir/Dialect/Math/IR/Math.h"   // IWYU pragma: keep (Passes.h.inc)
#include "mlir/IR/Iterators.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_FUSESINGLEQUBITUNITARYRUNS
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

/// Composed unitary and metadata for a fusable run.
struct FusableRunScan {
  Matrix2x2 composed = Matrix2x2::identity();
  std::size_t gateCount = 0;
  bool hasNonBasisGate = false;
  UnitaryOpInterface tail;
};

} // namespace

/// Whether `gate` has the structural shape of a fusable run member.
static bool isRunMemberCandidate(UnitaryOpInterface gate) {
  return gate && gate.isSingleQubit() && !isa<BarrierOp>(gate.getOperation());
}

/// Returns the matrix when `gate` can take part in a fusable
/// single-qubit run.
static std::optional<Matrix2x2> getRunMemberMatrix(UnitaryOpInterface gate) {
  if (!isRunMemberCandidate(gate)) {
    return std::nullopt;
  }
  return gate.getUnitaryMatrix<Matrix2x2>();
}

/// Walks the wire from @p head, composing the run's matrix and metadata.
///
/// @param head First gate of the run.
/// @param headMatrix Matrix already obtained while identifying the run head.
/// @param basis Single-qubit synthesis basis.
/// @return Composed matrix, gate count, and run tail.
static FusableRunScan
scanFusableRun(UnitaryOpInterface head, const Matrix2x2& headMatrix,
               const decomposition::SingleQubitBasis basis,
               const CompilerTarget* target) {
  FusableRunScan scan;
  for (auto* op : WireRange(head.getOutputQubit(0))) {
    auto member = dyn_cast_or_null<UnitaryOpInterface>(op);
    if (!member) {
      break;
    }
    const auto matrix = member.getOperation() == head.getOperation()
                            ? std::optional{headMatrix}
                            : getRunMemberMatrix(member);
    if (!matrix) {
      break;
    }
    scan.composed.premultiplyBy(*matrix);
    scan.hasNonBasisGate =
        scan.hasNonBasisGate ||
        (target != nullptr ? !target->supports(op)
                           : !decomposition::isSingleQubitBasisGate(op, basis));
    scan.tail = member;
    ++scan.gateCount;
  }
  return scan;
}

/// Erases a contiguous run from @p tail back to @p head.
///
/// @param rewriter The pattern rewriter.
/// @param head First gate of the run.
/// @param tail Last gate of the run.
static void eraseFusableRun(PatternRewriter& rewriter, UnitaryOpInterface head,
                            UnitaryOpInterface tail) {
  // Tail-first: each erased op is dead once its successor is gone.
  auto it = WireIterator(tail.getOutputQubit(0));
  auto* target = head.getOperation();
  while (*it != target) {
    auto* current = *it;
    --it;
    rewriter.eraseOp(current);
  }
  rewriter.eraseOp(target);
}

namespace {

/// Fuses maximal single-qubit unitary runs via Euler resynthesis.
struct FuseSingleQubitUnitaryRunsPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  FuseSingleQubitUnitaryRunsPattern(
      MLIRContext* context, const decomposition::SingleQubitBasis basis,
      decomposition::SingleQubitFusionPolicy policy,
      const CompilerTarget* target)
      : OpInterfaceRewritePattern(context), basis(basis), policy(policy),
        target(target) {}

  decomposition::SingleQubitBasis basis;
  decomposition::SingleQubitFusionPolicy policy;
  const CompilerTarget* target;

  /// Fuses the run anchored at `op` when beneficial.
  ///
  /// Fuses if the run contains a non-basis gate or Euler resynthesis would
  /// shorten it (@ref synthesizeUnitary1QEuler).
  ///
  /// @param op The matched unitary operation.
  /// @param rewriter The pattern rewriter.
  /// @return `success()` if a run was fused, `failure()` otherwise.
  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    if (policy.skipControlledBodies &&
        (op.getOperation()->getParentOfType<CtrlOp>() != nullptr)) {
      return failure();
    }
    if (!isRunMemberCandidate(op)) {
      return failure();
    }
    if (policy.preserveSingletons &&
        !isRunMemberCandidate(
            dyn_cast<UnitaryOpInterface>(*op.getOutputQubit(0).user_begin()))) {
      return failure();
    }
    auto predecessor = dyn_cast_or_null<UnitaryOpInterface>(
        op.getInputQubit(0).getDefiningOp());
    if (getRunMemberMatrix(predecessor)) {
      return failure();
    }
    const auto headMatrix = getRunMemberMatrix(op);
    if (!headMatrix) {
      return failure();
    }

    FusableRunScan run = scanFusableRun(op, *headMatrix, basis, target);
    if (policy.preserveSingletons && run.gateCount == 1) {
      return failure();
    }
    const auto synthesized = decomposition::synthesizeUnitary1QEuler(
        rewriter, op.getLoc(), op.getInputQubit(0), run.composed, run.gateCount,
        run.hasNonBasisGate, basis);
    if (!synthesized) {
      return failure();
    }
    decomposition::emitGPhaseIfNeeded(rewriter, op.getLoc(),
                                      synthesized->globalPhase);

    rewriter.replaceAllUsesWith(run.tail.getOutputQubit(0), synthesized->qubit);
    eraseFusableRun(rewriter, op, run.tail);
    return success();
  }
};

/// Pass that fuses single-qubit unitary runs via Euler resynthesis.
struct FuseSingleQubitUnitaryRunsPass final
    : impl::FuseSingleQubitUnitaryRunsBase<FuseSingleQubitUnitaryRunsPass> {
  using Base::Base;

  explicit FuseSingleQubitUnitaryRunsPass(
      FuseSingleQubitUnitaryRunsOptions options)
      : Base(std::move(options)) {}

  explicit FuseSingleQubitUnitaryRunsPass(const CompilerTarget& target)
      : target_(target) {}

protected:
  void runOnOperation() override {
    auto moduleOp = getOperation();

    auto parsed = decomposition::parseSingleQubitBasis(basis);
    decomposition::SingleQubitFusionPolicy policy;
    if (target_) {
      const auto nativeBasis = target_->synthesisBasis();
      if (!nativeBasis) {
        return;
      }
      parsed = nativeBasis->singleQubit;
      policy = decomposition::SingleQubitFusionPolicy::forTarget(*parsed);
    }
    if (!parsed) {
      moduleOp.emitError()
          << "Invalid single-qubit synthesis basis '" << basis
          << "'. Expected one of: zyz, zxz, xzx, xyx, u, zsxx, r.";
      signalPassFailure();
      return;
    }

    const auto* target = target_ ? &*target_ : nullptr;
    if (failed(decomposition::fuseSingleQubitUnitaryRuns(
            moduleOp, *parsed, policy, target, GreedyRewriteConfig{})) ||
        failed(mlir::mqt::normalizeGlobalPhases(moduleOp))) {
      moduleOp.emitError("fusion pipeline failed"); // LCOV_EXCL_LINE
      signalPassFailure();
    }
  }

private:
  std::optional<CompilerTarget> target_;
};

} // namespace

std::unique_ptr<Pass>
createFuseSingleQubitUnitaryRuns(const CompilerTarget& target) {
  return std::make_unique<FuseSingleQubitUnitaryRunsPass>(target);
}

} // namespace mlir::qco

namespace mlir::qco::decomposition {

LogicalResult fuseSingleQubitUnitaryRuns(ModuleOp moduleOp,
                                         SingleQubitBasis basis,
                                         SingleQubitFusionPolicy policy,
                                         const CompilerTarget* target,
                                         const GreedyRewriteConfig& config) {
  SingleQubitRunFusion fusion(basis, policy, target, config);
  return failure(moduleOp
                     ->walk<WalkOrder::PostOrder, ReverseIterator>(
                         [&](Operation* operation) {
                           return failed(fusion.apply(operation))
                                      ? WalkResult::interrupt()
                                      : WalkResult::advance();
                         })
                     .wasInterrupted());
}

SingleQubitRunFusion::SingleQubitRunFusion(SingleQubitBasis basis,
                                           SingleQubitFusionPolicy policy,
                                           const CompilerTarget* target,
                                           GreedyRewriteConfig config)
    : basis_(basis), policy_(policy), target_(target), config_(config) {
  // Do not rewrite producers or unrelated runs during the caller's walk.
  config_.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
}

LogicalResult SingleQubitRunFusion::apply(Operation* operation) {
  auto head = dyn_cast<UnitaryOpInterface>(operation);
  if (!isRunMemberCandidate(head) ||
      (policy_.skipControlledBodies && operation->getParentOfType<CtrlOp>())) {
    return success();
  }
  Value input = head.getInputQubit(0);
  if (isRunMemberCandidate(
          dyn_cast_or_null<UnitaryOpInterface>(input.getDefiningOp())) ||
      (policy_.preserveSingletons &&
       !isRunMemberCandidate(dyn_cast<UnitaryOpInterface>(
           *head.getOutputQubit(0).user_begin())))) {
    return success();
  }
  bool hasRuntimeParameters = false;
  SmallVector<Operation*> members;
  SmallVector<Operation*> candidates;
  const auto collectCandidates = [&] {
    candidates.clear();
    auto start = dyn_cast<UnitaryOpInterface>(*input.user_begin());
    if (!isRunMemberCandidate(start)) {
      return;
    }
    bool predecessorHasMatrix = false;
    for (auto* current : WireRange(start.getOutputQubit(0))) {
      auto op = dyn_cast<UnitaryOpInterface>(current);
      if (!isRunMemberCandidate(op)) {
        break;
      }
      members.push_back(current);
      hasRuntimeParameters =
          hasRuntimeParameters ||
          (canSynthesizeParameterizedUnitary1Q(op.getOperation()) &&
           llvm::any_of(op.getParameters(), [](Value parameter) {
             return !mqt::valueToConstantDouble(parameter);
           }));
      if ((!policy_.preserveSingletons ||
           isRunMemberCandidate(dyn_cast<UnitaryOpInterface>(
               *op.getOutputQubit(0).user_begin()))) &&
          !predecessorHasMatrix) {
        candidates.push_back(current);
      }
      predecessorHasMatrix = getRunMemberMatrix(op).has_value();
    }
  };
  collectCandidates();
  if (candidates.empty()) {
    return success();
  }
  if (hasRuntimeParameters) {
    if (!runtimePatterns_) {
      RewritePatternSet patterns(operation->getContext());
      populateParameterizedSingleQubitRunCompositionPatterns(patterns, basis_,
                                                             policy_, target_);
      runtimePatterns_.emplace(std::move(patterns));
    }
    if (failed(applyOpPatternsGreedily(members, *runtimePatterns_, config_))) {
      return failure();
    }
    // Runtime rewrites can replace the collected operations.
    members.clear();
    collectCandidates();
  }
  if (candidates.empty()) {
    return success();
  }
  if (!matrixPatterns_) {
    RewritePatternSet patterns(input.getContext());
    populateFuseSingleQubitUnitaryRunsPatterns(patterns, basis_, policy_,
                                               target_);
    matrixPatterns_.emplace(std::move(patterns));
  }
  return applyOpPatternsGreedily(candidates, *matrixPatterns_, config_);
}

void populateFuseSingleQubitUnitaryRunsPatterns(RewritePatternSet& patterns,
                                                const SingleQubitBasis basis,
                                                SingleQubitFusionPolicy policy,
                                                const CompilerTarget* target) {
  patterns.add<FuseSingleQubitUnitaryRunsPattern>(patterns.getContext(), basis,
                                                  policy, target);
}

} // namespace mlir::qco::decomposition
