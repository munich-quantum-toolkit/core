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
#include "mlir/Dialect/QCO/IR/QCOUnitaryMatrixInterfaces.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"

#include <Eigen/Core>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h> // IWYU pragma: keep (Passes.h.inc)
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <iterator>
#include <optional>
#include <ranges>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_FUSESINGLEQUBITUNITARYRUNS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

/**
 * @brief Whether `op` lives inside an `inv`/`ctrl` modifier body.
 *
 * A modifier exposes its body's combined unitary through `getUnitaryMatrix` and
 * is fused as a single, atomic run member (when single-qubit). Fusing the gates
 * inside its body would invalidate that matrix, so body gates are never run
 * members themselves.
 *
 * @param op The operation to test.
 * @return `true` if `op`'s parent is an `inv` or `ctrl` op.
 */
static bool isNestedInModifierRegion(Operation* op) {
  Operation* parent = op->getParentOp();
  return parent != nullptr && isa<InvOp, CtrlOp>(parent);
}

/**
 * @brief Whether `op` may participate in a fusable single-qubit run.
 *
 * @param op The unitary operation to test.
 * @return `true` for a single-qubit, matrix-backed unitary that lives directly
 *         on a wire (not inside a modifier body).
 */
static bool isFuseCandidate(UnitaryOpInterface op) {
  if (!op || !op.isSingleQubit() || isNestedInModifierRegion(op)) {
    return false;
  }
  return isa<UnitaryMatrixOpInterface>(op.getOperation());
}

/**
 * @brief Returns the compile-time 2x2 unitary matrix of `op`, if available.
 *
 * @param op The unitary operation to query.
 * @return The constant matrix, or `std::nullopt` if `op` is not matrix-backed
 * or its matrix is not known at compile time.
 */
static std::optional<Eigen::Matrix2cd> getConstMatrix(UnitaryOpInterface op) {
  auto matrixOp = dyn_cast<UnitaryMatrixOpInterface>(op.getOperation());
  if (!matrixOp) {
    return std::nullopt;
  }
  Eigen::Matrix2cd m;
  if (!matrixOp.getUnitaryMatrix2x2(m)) {
    return std::nullopt;
  }
  return m;
}

/**
 * @brief Whether `op` can participate in a fusable run.
 *
 * @param op The operation to test.
 * @return `true` for a single-qubit, matrix-backed unitary outside a modifier
 *         body whose matrix is known at compile time.
 */
static bool isRunMember(Operation* op) {
  auto iface = dyn_cast<UnitaryOpInterface>(op);
  return iface && isFuseCandidate(iface) && getConstMatrix(iface).has_value();
}

/**
 * @brief Composes a run of unitary ops into a single matrix.
 *
 * @param run The run members in execution (circuit) order.
 * @return The product of the members' matrices.
 */
static Eigen::Matrix2cd composeRun(ArrayRef<UnitaryOpInterface> run) {
  Eigen::Matrix2cd composed = Eigen::Matrix2cd::Identity();
  for (auto op : run) {
    // Execution order: first op applied first => multiply on the left.
    composed = (*getConstMatrix(op)) * composed;
  }
  return composed;
}

/**
 * @brief Whether `op` is one of the gates the target `basis` emits.
 *
 * The gate sets mirror `emitKAK` and `emitFromPSXSequence` in `Euler.cpp`. The
 * greedy driver re-visits the gates produced by a rewrite, so this lets the
 * pattern detect a run that is already expressed entirely in the target basis
 * and avoid re-fusing the gates it just produced.
 *
 * @param op The operation to classify.
 * @param basis The target Euler basis.
 * @return `true` if `op` is a gate the `basis` emits.
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

namespace {

/**
 * @brief Replaces a maximal single-qubit unitary run with its Euler synthesis.
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
   * A run head is a fusable op whose predecessor on the wire is not itself a
   * fusable run member, so each run is matched exactly once at its start.
   *
   * @param op The candidate run head.
   * @return `true` if `op` starts a run.
   */
  static bool isRunStart(UnitaryOpInterface op) {
    if (!isRunMember(op.getOperation())) {
      return false;
    }
    Operation* pred = op.getInputTarget(0).getDefiningOp();
    return pred == nullptr || !isRunMember(pred);
  }

  /**
   * @brief Collects the maximal run of fusable ops starting at `start`.
   *
   * Follows the wire forward while staying within the same block.
   *
   * @param start The run head (must satisfy `isRunStart`).
   * @return The run members in circuit order.
   */
  static SmallVector<UnitaryOpInterface> collectRun(UnitaryOpInterface start) {
    SmallVector<UnitaryOpInterface> run{start};
    Block* block = start->getBlock();
    for (WireIterator it = std::next(WireIterator(start.getOutputTarget(0)));
         it != std::default_sentinel; ++it) {
      Operation* op = it.operation();
      if (op->getBlock() != block || !isRunMember(op)) {
        break;
      }
      run.emplace_back(cast<UnitaryOpInterface>(op));
    }
    return run;
  }

  /**
   * @brief Fuses the run anchored at `op` into its Euler resynthesis.
   *
   * @param op The matched unitary operation.
   * @param rewriter Pattern rewriter used to apply the transformation.
   * @return `success()` if a run was fused, `failure()` otherwise.
   */
  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    if (!isRunStart(op)) {
      return failure();
    }

    auto run = collectRun(op);
    const Eigen::Matrix2cd composed = composeRun(run);

    // Resynthesize a run when it either contains a gate outside the target
    // basis (so it is not yet expressed in the native gate set) or is already
    // in-basis but longer than the canonical Euler form (so fusing shortens
    // it). A run that is in-basis and already at canonical length is left
    // untouched, which is also what keeps the greedy driver from re-matching
    // the gates this pattern just produced.
    const bool hasNonBasisGate =
        llvm::any_of(run, [&](UnitaryOpInterface member) {
          return !isTargetBasisGate(member.getOperation(), basis);
        });
    if (!hasNonBasisGate &&
        run.size() <= decomposition::synthesisGateCount(composed, basis)) {
      return failure();
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op.getOperation());
    const Value qubit = decomposition::synthesizeUnitary1QEuler(
        rewriter, op.getLoc(), op.getInputTarget(0), composed, basis);

    rewriter.replaceAllUsesWith(run.back().getOutputTarget(0), qubit);
    for (UnitaryOpInterface member : std::ranges::reverse_view(run)) {
      rewriter.eraseOp(member.getOperation());
    }
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

    const auto parsed = decomposition::parseEulerBasis(this->basis);
    if (!parsed) {
      module.emitError() << "Invalid Euler basis '" << this->basis
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
