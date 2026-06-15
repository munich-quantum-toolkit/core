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
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cassert>
#include <iterator>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_FUSESINGLEQUBITUNITARYRUNS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

/**
 * @brief Whether `op` has a compile-time 2x2 unitary without synthesizing it.
 *
 * For parameterized gates, only checks that operands are `arith.constant`
 * scalars. Other static single-qubit gates need no matrix query. `inv` is the
 * only parameter-free fuse candidate that may still lack a compile-time matrix.
 *
 * @param op The unitary operation to test.
 * @return `true` when `getUnitaryMatrix<Matrix2x2>()` would succeed.
 */
static bool hasCompileTimeUnitaryMatrix2x2(UnitaryOpInterface op) {
  for (size_t i = 0; i < op.getNumParams(); ++i) {
    if (!mlir::utils::valueToDouble(op.getParameter(i))) {
      return false;
    }
  }
  if (op.getNumParams() > 0 || !isa<InvOp>(op.getOperation())) {
    return true;
  }
  Matrix2x2 unused;
  return op.getUnitaryMatrix2x2(unused);
}

/**
 * @brief Whether `op` is a fusable single-qubit run member on the wire.
 *
 * @param op The unitary operation to test.
 * @return `true` for a single-qubit unitary with a compile-time 2x2 matrix,
 *         excluding barriers.
 */
static bool isRunMember(UnitaryOpInterface op) {
  return op && op.isSingleQubit() && !isa<BarrierOp>(op.getOperation()) &&
         hasCompileTimeUnitaryMatrix2x2(op);
}

/**
 * @brief Whether `op` is a fusable single-qubit run member on the wire.
 *
 * @param op The operation to test.
 * @return `true` for a single-qubit unitary with a compile-time 2x2 matrix,
 *         excluding barriers.
 */
static bool isRunMember(Operation* op) {
  return isRunMember(dyn_cast<UnitaryOpInterface>(op));
}

/**
 * @brief Whether @p op sits in the body region of an `inv`.
 *
 * Run heads are not started inside `inv` bodies; the enclosing `InvOp` already
 * exposes the composed body unitary on the parent wire.
 */
static bool isInsideInvBody(Operation* op) {
  return op != nullptr && op->getParentOfType<InvOp>() != nullptr;
}

/**
 * @brief Whether `op` is a gate the target `basis` emits.
 *
 * Gate sets match the synthesis step kinds in `Euler.cpp`. Used to
 * skip runs that are already in the target basis at canonical length.
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
 * @brief Composed unitary and metadata for a fusable run, without storing ops.
 */
struct FusableRunScan {
  Matrix2x2 composed = Matrix2x2::identity();
  std::size_t gateCount = 0;
  bool hasNonBasisGate = false;
  UnitaryOpInterface tail;
};

/**
 * @brief Walks the wire from @p head, composing matrices and run metadata.
 *
 * `WireIterator` stops at region boundaries, so members are consecutive on the
 * wire. Only @p tail is retained for replacement and erasure.
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
   * @brief Whether `op` is the head of a run.
   *
   * @param op The candidate run head.
   * @return `true` if `op` is a run member outside any `inv` body whose wire
   *         predecessor is not a run member.
   */
  static bool isRunStart(UnitaryOpInterface op) {
    if (!isRunMember(op)) {
      return false;
    }
    if (isInsideInvBody(op.getOperation())) {
      return false;
    }
    Operation* pred = op.getInputTarget(0).getDefiningOp();
    return pred == nullptr || !isRunMember(pred);
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
