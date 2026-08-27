/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"

#include <mlir/Dialect/Arith/IR/Arith.h> // IWYU pragma: keep (Passes.h.inc)
#include <mlir/Dialect/Math/IR/Math.h>   // IWYU pragma: keep (Passes.h.inc)
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cstddef>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_FUSESINGLEQUBITUNITARYRUNS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

/** Composed unitary and metadata for a fusable run. */
struct FusableRunScan {
  Matrix2x2 composed = Matrix2x2::identity();
  std::size_t gateCount = 0;
  bool hasNonBasisGate = false;
  UnitaryOpInterface tail;
};

} // namespace

/**
 * @brief Whether `gate` has the structural shape of a fusable run member.
 */
static bool isRunMemberCandidate(UnitaryOpInterface gate) {
  return gate && gate.isSingleQubit() && !isa<BarrierOp>(gate.getOperation());
}

/**
 * @brief Returns the matrix when `gate` can take part in a fusable
 * single-qubit run.
 */
static std::optional<Matrix2x2> getRunMemberMatrix(UnitaryOpInterface gate) {
  if (!isRunMemberCandidate(gate)) {
    return std::nullopt;
  }
  return gate.getUnitaryMatrix<Matrix2x2>();
}

/**
 * @brief Walks the wire from @p head, composing the run's matrix and metadata.
 *
 * @param head First gate of the run.
 * @param headMatrix Matrix already obtained while identifying the run head.
 * @param basis Single-qubit synthesis basis.
 * @return Composed matrix, gate count, and run tail.
 */
static FusableRunScan
scanFusableRun(UnitaryOpInterface head, const Matrix2x2& headMatrix,
               const decomposition::SingleQubitBasis basis) {
  FusableRunScan scan;
  for (auto* op : WireRange(head.getOutputTarget(0))) {
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
    scan.hasNonBasisGate |= !decomposition::isSingleQubitBasisGate(op, basis);
    scan.tail = member;
    ++scan.gateCount;
  }
  return scan;
}

/**
 * @brief Erases a contiguous run from @p tail back to @p head.
 *
 * @param rewriter The pattern rewriter.
 * @param head First gate of the run.
 * @param tail Last gate of the run.
 */
static void eraseFusableRun(PatternRewriter& rewriter, UnitaryOpInterface head,
                            UnitaryOpInterface tail) {
  // Tail-first: each erased op is dead once its successor is gone.
  auto it = WireIterator(tail.getOutputTarget(0));
  auto* target = head.getOperation();
  while (*it != target) {
    auto* current = *it;
    --it;
    rewriter.eraseOp(current);
  }
  rewriter.eraseOp(target);
}

namespace {

/**
 * @brief Fuses maximal single-qubit unitary runs via Euler resynthesis.
 */
struct FuseSingleQubitUnitaryRunsPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  FuseSingleQubitUnitaryRunsPattern(MLIRContext* context,
                                    const decomposition::SingleQubitBasis basis,
                                    const bool skipControlledBodies)
      : OpInterfaceRewritePattern(context), basis(basis),
        skipControlledBodies(skipControlledBodies) {}

  decomposition::SingleQubitBasis basis;
  bool skipControlledBodies;

  /**
   * @brief Fuses the run anchored at `op` when beneficial.
   *
   * Fuses if the run contains a non-basis gate or Euler resynthesis would
   * shorten it (@ref synthesizeUnitary1QEuler).
   *
   * @param op The matched unitary operation.
   * @param rewriter The pattern rewriter.
   * @return `success()` if a run was fused, `failure()` otherwise.
   */
  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    if (skipControlledBodies &&
        (op.getOperation()->getParentOfType<CtrlOp>() != nullptr)) {
      return failure();
    }
    if (!isRunMemberCandidate(op)) {
      return failure();
    }
    auto predecessor = dyn_cast_or_null<UnitaryOpInterface>(
        op.getInputTarget(0).getDefiningOp());
    if (getRunMemberMatrix(predecessor)) {
      return failure();
    }
    const auto headMatrix = getRunMemberMatrix(op);
    if (!headMatrix) {
      return failure();
    }

    FusableRunScan run = scanFusableRun(op, *headMatrix, basis);
    const auto synthesized = decomposition::synthesizeUnitary1QEuler(
        rewriter, op.getLoc(), op.getInputTarget(0), run.composed,
        run.gateCount, run.hasNonBasisGate, basis);
    if (!synthesized) {
      return failure();
    }
    decomposition::emitGPhaseIfNeeded(rewriter, op.getLoc(),
                                      synthesized->globalPhase);

    rewriter.replaceAllUsesWith(run.tail.getOutputTarget(0),
                                synthesized->qubit);
    eraseFusableRun(rewriter, op, run.tail);
    return success();
  }
};

/**
 * @brief Pass that fuses single-qubit unitary runs via Euler resynthesis.
 */
struct FuseSingleQubitUnitaryRunsPass final
    : impl::FuseSingleQubitUnitaryRunsBase<FuseSingleQubitUnitaryRunsPass> {
  using Base::Base;

  explicit FuseSingleQubitUnitaryRunsPass(
      FuseSingleQubitUnitaryRunsOptions options)
      : Base(std::move(options)) {}

protected:
  void runOnOperation() override {
    auto moduleOp = getOperation();

    const auto parsed = decomposition::parseSingleQubitBasis(basis);
    if (!parsed) {
      moduleOp.emitError()
          << "Invalid single-qubit synthesis basis '" << basis
          << "'. Expected one of: zyz, zxz, xzx, xyx, u, zsxx, r.";
      signalPassFailure();
      return;
    }

    RewritePatternSet compositionPatterns(&getContext());
    decomposition::populateParameterizedSingleQubitRunCompositionPatterns(
        compositionPatterns, *parsed);

    RewritePatternSet patterns(&getContext());
    decomposition::populateFuseSingleQubitUnitaryRunsPatterns(
        patterns, *parsed, /*skipControlledBodies=*/false);

    if (failed(
            applyPatternsGreedily(moduleOp, std::move(compositionPatterns))) ||
        failed(applyPatternsGreedily(moduleOp, std::move(patterns))) ||
        failed(mlir::mqt::normalizeGlobalPhases(moduleOp))) {
      moduleOp.emitError("fusion pipeline failed"); // LCOV_EXCL_LINE
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco

namespace mlir::qco::decomposition {

void populateFuseSingleQubitUnitaryRunsPatterns(
    RewritePatternSet& patterns, const SingleQubitBasis basis,
    const bool skipControlledBodies) {
  patterns.add<FuseSingleQubitUnitaryRunsPattern>(patterns.getContext(), basis,
                                                  skipControlledBodies);
}

} // namespace mlir::qco::decomposition
