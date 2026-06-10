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
 * @brief Whether `op` is inside an `inv`/`ctrl` body.
 *
 * The modifier's combined unitary is fused as one run member; gates inside its
 * body are not separate run members.
 *
 * @param op The operation to test.
 * @return `true` if any ancestor is `inv` or `ctrl`.
 */
static bool isNestedInModifierRegion(Operation* op) {
  return op != nullptr && (op->getParentOfType<InvOp>() != nullptr ||
                           op->getParentOfType<CtrlOp>() != nullptr);
}

/**
 * @brief Whether `op` may participate in a fusable single-qubit run.
 *
 * @param op The unitary operation to test.
 * @return `true` for a single-qubit, matrix-backed unitary on the wire, outside
 *         a modifier body.
 */
static bool isFuseCandidate(UnitaryOpInterface op) {
  if (!op || !op.isSingleQubit() || isNestedInModifierRegion(op) ||
      isa<BarrierOp>(op.getOperation())) {
    return false;
  }
  Matrix2x2 matrix;
  return op.getUnitaryMatrix2x2(matrix);
}

/**
 * @brief Returns the compile-time 2x2 unitary matrix of `op`, if available.
 *
 * @param op The unitary operation to query.
 * @return The matrix, or `std::nullopt` if not known at compile time.
 */
static std::optional<Matrix2x2> getConstMatrix(UnitaryOpInterface op) {
  Matrix2x2 matrix;
  if (!op.getUnitaryMatrix2x2(matrix)) {
    return std::nullopt;
  }
  return matrix;
}

/**
 * @brief Whether `op` can participate in a fusable run.
 *
 * @param op The operation to test.
 * @return `true` for a fuse candidate with a known compile-time matrix.
 */
static bool isRunMember(Operation* op) {
  auto iface = dyn_cast<UnitaryOpInterface>(op);
  return iface && isFuseCandidate(iface) && getConstMatrix(iface).has_value();
}

/**
 * @brief Composes a run of unitary ops into a single matrix.
 *
 * @param run The run members in circuit order.
 * @return The product of their matrices.
 */
static Matrix2x2 composeRun(ArrayRef<UnitaryOpInterface> run) {
  Matrix2x2 composed = Matrix2x2::identity();
  for (auto op : run) {
    // First gate in the run is applied first (left factor).
    composed = (*getConstMatrix(op)) * composed;
  }
  return composed;
}

/**
 * @brief Whether `op` is a gate the target `basis` emits.
 *
 * Gate sets match `emitKAK` and `emitFromPSXSequence` in `Euler.cpp`. Used to
 * skip runs that are already in the target basis at canonical length.
 *
 * @param op The operation to classify.
 * @param basis The target Euler basis.
 * @return `true` if `op` is emitted by synthesis in `basis`.
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
   * @return `true` if the wire predecessor is not a run member.
   */
  static bool isRunStart(UnitaryOpInterface op) {
    if (!isRunMember(op.getOperation())) {
      return false;
    }
    Operation* pred = op.getInputTarget(0).getDefiningOp();
    return pred == nullptr || !isRunMember(pred);
  }

  /**
   * @brief Collects the maximal fusable run starting at `start`.
   *
   * @param start The run head.
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
   * @brief Fuses the run anchored at `op` when beneficial.
   *
   * Fuses if the run contains a non-basis gate or is longer than the canonical
   * synthesis for its composed matrix.
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

    auto run = collectRun(op);
    const Matrix2x2 composed = composeRun(run);
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
