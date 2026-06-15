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
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h> // IWYU pragma: keep (Passes.h.inc)
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cassert>
#include <cstddef>
#include <iterator>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_FUSESINGLEQUBITUNITARYRUNS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

/**
 * @brief Composed unitary and metadata for a fusable run, without storing ops.
 */
struct FusableRunScan {
  Matrix2x2 composed = Matrix2x2::identity();
  std::size_t gateCount = 0;
  bool hasNonBasisGate = false;
  UnitaryOpInterface tail;
};

} // namespace

/**
 * @brief Whether `op` can take part in a fusable single-qubit run.
 *
 * A run member is a non-barrier, single-qubit unitary whose 2x2 matrix is known
 * at compile time. Parameterized gates only need constant parameters (no matrix
 * is built); `inv` is the only parameter-free op that may still lack a constant
 * matrix, so it is queried directly. An `inv` that hides a barrier in its body
 * is rejected: its 2x2 matrix ignores the barrier, so absorbing the modifier
 * would silently drop it. Such bodies instead fuse around the barrier in place.
 *
 * @param op The operation to test. May be null (e.g. a missing predecessor).
 * @return `true` for a non-barrier single-qubit unitary with a compile-time 2x2
 *         matrix that does not hide a barrier inside an `inv` body.
 */
static bool isRunMember(Operation* op) {
  auto gate = dyn_cast_or_null<UnitaryOpInterface>(op);
  if (!gate || !gate.isSingleQubit() || isa<BarrierOp>(op)) {
    return false;
  }
  for (size_t i = 0; i < gate.getNumParams(); ++i) {
    if (!mlir::utils::valueToDouble(gate.getParameter(i))) {
      return false;
    }
  }
  if (gate.getNumParams() > 0 || !isa<InvOp>(op)) {
    return true;
  }
  const bool hidesBarrier = op->walk([](BarrierOp) {
                                return WalkResult::interrupt();
                              }).wasInterrupted();
  Matrix2x2 unused;
  return !hidesBarrier && gate.getUnitaryMatrix2x2(unused);
}

/**
 * @brief Whether `op` is a gate that Euler synthesis emits for `basis`.
 *
 * Mirrors the synthesis step kinds in `Euler.cpp`; used to detect runs that are
 * already in the target basis at canonical length.
 *
 * @param op The operation to classify.
 * @param basis The target Euler basis.
 * @return `true` if `op` is in the gate set Euler synthesis emits for `basis`.
 */
static bool isTargetBasisGate(Operation* op, decomposition::EulerBasis basis) {
  using decomposition::EulerBasis;
  return TypeSwitch<Operation*, bool>(op)
      .Case<RZOp>([&](auto) {
        return basis == EulerBasis::ZYZ || basis == EulerBasis::ZXZ ||
               basis == EulerBasis::XZX || basis == EulerBasis::ZSXX;
      })
      .Case<RYOp>([&](auto) {
        return basis == EulerBasis::ZYZ || basis == EulerBasis::XYX;
      })
      .Case<RXOp>([&](auto) {
        return basis == EulerBasis::ZXZ || basis == EulerBasis::XZX ||
               basis == EulerBasis::XYX;
      })
      .Case<UOp>([&](auto) { return basis == EulerBasis::U; })
      .Case<SXOp, XOp>([&](auto) { return basis == EulerBasis::ZSXX; })
      .Default([](auto) { return false; });
}

/**
 * @brief Walks the wire from @p head, composing the run's matrix and metadata.
 *
 * `WireIterator` stops at region boundaries, so run members are consecutive on
 * the wire. Only the run tail is retained, for replacement and erasure.
 *
 * @param head First gate of the run.
 * @param basis Target Euler basis (for non-basis detection).
 * @return Composed matrix, gate count, and run tail.
 */
static FusableRunScan scanFusableRun(UnitaryOpInterface head,
                                     decomposition::EulerBasis basis) {
  FusableRunScan scan;
  const auto accumulate = [&](UnitaryOpInterface member) {
    const auto matrix = member.getUnitaryMatrix<Matrix2x2>();
    assert(matrix && "run member must have a compile-time 2x2 matrix");
    scan.composed.premultiplyBy(*matrix);
    scan.hasNonBasisGate |= !isTargetBasisGate(member.getOperation(), basis);
    scan.tail = member;
    ++scan.gateCount;
  };

  accumulate(head);
  for (WireIterator it = std::next(WireIterator(head.getOutputTarget(0)));
       it != std::default_sentinel; ++it) {
    Operation* memberOp = it.operation();
    if (!isRunMember(memberOp)) {
      break;
    }
    accumulate(cast<UnitaryOpInterface>(memberOp));
  }
  return scan;
}

/**
 * @brief Erases a contiguous run from tail back to @p head.
 */
static void eraseFusableRun(PatternRewriter& rewriter, UnitaryOpInterface head,
                            UnitaryOpInterface tail) {
  // Erase tail-first so each op is dead (its successor is already gone) when
  // removed; capture the predecessor before erasing the current op.
  UnitaryOpInterface current = tail;
  while (current.getOperation() != head.getOperation()) {
    auto pred =
        cast<UnitaryOpInterface>(current.getInputTarget(0).getDefiningOp());
    rewriter.eraseOp(current.getOperation());
    current = pred;
  }
  rewriter.eraseOp(head.getOperation());
}

namespace {

/**
 * @brief Fuses maximal single-qubit unitary runs via Euler resynthesis.
 *
 * Matches at each run head so each run is rewritten once.
 */
struct FuseSingleQubitUnitaryRunsPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  FuseSingleQubitUnitaryRunsPattern(MLIRContext* context,
                                    decomposition::EulerBasis basis)
      : OpInterfaceRewritePattern(context), basis(basis) {}

  decomposition::EulerBasis basis;

  /**
   * @brief Whether `op` starts a run.
   *
   * A run does not start inside the body of a single-qubit `inv`: that modifier
   * is itself a run member, so the run on the parent wire absorbs the whole
   * `inv` as one unitary and fusing its body in place would be redundant. A
   * multi-qubit `inv` cannot be absorbed that way (it has no compile-time 2x2
   * matrix), so single-qubit chains inside its body are fused locally and may
   * start runs.
   *
   * @param op The candidate run head.
   * @return `true` if `op` is a run member whose wire predecessor is not itself
   * a run member, and which is not inside the body of a fusable single-qubit
   * `inv`.
   */
  static bool isRunStart(UnitaryOpInterface op) {
    if (!isRunMember(op.getOperation())) {
      return false;
    }
    // A single-qubit `inv` is itself a run member, so the parent wire's run
    // already absorbs the whole modifier; only the bodies of multi-qubit `inv`
    // modifiers (which cannot be absorbed) host their own runs.
    if (auto inv = op->getParentOfType<InvOp>();
        inv && isRunMember(inv.getOperation())) {
      return false;
    }
    return !isRunMember(op.getInputTarget(0).getDefiningOp());
  }

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
    if (!isRunStart(op)) {
      return failure();
    }

    FusableRunScan run = scanFusableRun(op, basis);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op.getOperation());
    const std::optional<Value> qubitOut =
        decomposition::synthesizeUnitary1QEuler(
            rewriter, op.getLoc(), op.getInputTarget(0), run.composed,
            run.gateCount, run.hasNonBasisGate, basis);
    if (!qubitOut) {
      return failure();
    }

    rewriter.replaceAllUsesWith(run.tail.getOutputTarget(0), *qubitOut);
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
    auto module = getOperation();

    const auto parsed = decomposition::parseEulerBasis(basis);
    if (!parsed) {
      module.emitError() << "Invalid Euler basis '" << basis
                         << "'. Expected one of: zyz, zxz, xzx, xyx, u, zsxx.";
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<FuseSingleQubitUnitaryRunsPattern>(patterns.getContext(),
                                                    *parsed);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco
