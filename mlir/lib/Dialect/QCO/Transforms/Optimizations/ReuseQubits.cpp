/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cassert>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_REUSEQUBITS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {
class ReuseAnalysis {
public:
  [[nodiscard]] static std::optional<ReuseAnalysis> analyze(AllocOp alloc) {
    ReuseAnalysis analysis(alloc->getBlock());
    getForwardSlice(alloc.getResult(), &analysis.forwardSlice);

    for (auto* operation : analysis.forwardSlice) {
      auto* ancestor = analysis.block->findAncestorOpInBlock(*operation);
      if (ancestor == nullptr) {
        return std::nullopt;
      }
      analysis.users.insert(ancestor);
    }

    for (auto& operation : *analysis.block) {
      if (!analysis.users.contains(&operation) || isa<SinkOp>(operation) ||
          isMemoryEffectFree(&operation)) {
        continue;
      }
      analysis.firstEffectfulUser = &operation;
      break;
    }
    return analysis;
  }

  [[nodiscard]] bool canReuse(SinkOp sink) const {
    return !forwardSlice.contains(sink.getOperation()) &&
           (firstEffectfulUser == nullptr ||
            sink->isBeforeInBlock(firstEffectfulUser));
  }

  void moveUsersAfter(Operation* insertionPoint,
                      PatternRewriter& rewriter) const {
    assert(insertionPoint->getBlock() == block &&
           "reuse point must be in the analyzed block");

    SmallVector<Operation*> operationsToMove;
    for (auto& operation : *block) {
      if (&operation == insertionPoint) {
        break;
      }
      if (users.contains(&operation)) {
        operationsToMove.push_back(&operation);
      }
    }

    for (auto* operation : operationsToMove) {
      rewriter.moveOpAfter(operation, insertionPoint);
      insertionPoint = operation;
    }
  }

private:
  explicit ReuseAnalysis(Block* const block) : block(block) {}

  Block* block;
  SetVector<Operation*> forwardSlice;
  llvm::DenseSet<Operation*> users;
  Operation* firstEffectfulUser = nullptr;
};

/**
 * @brief This is the main qubit reuse pattern.
 */
struct ReuseQubitsPattern final : OpRewritePattern<AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  /**
   * @brief Rewrites the given `AllocOp` and `SinkOp` to reuse the
   * qubit instead.
   *
   * @param alloc The allocation that will be replaced by qubit reuse.
   * @param sink The sink that will be replaced by a new reset
   * operation.
   * @param rewriter The pattern rewriter to use for the rewrite.
   */
  static void rewriteForReuse(AllocOp alloc, SinkOp sink,
                              const ReuseAnalysis& analysis,
                              PatternRewriter& rewriter) {
    rewriter.setInsertionPointAfter(sink);
    auto reset = rewriter.replaceOpWithNewOp<ResetOp>(
        alloc, alloc.getResult().getType(), sink.getQubit());
    rewriter.eraseOp(sink);

    analysis.moveUsersAfter(reset, rewriter);
  }

  LogicalResult matchAndRewrite(AllocOp op,
                                PatternRewriter& rewriter) const override {
    // Find all `SinkOp` operations in the current block and check
    // if any of them are disjoint from the qubit being allocated, indicating
    // potential for reuse.

    const auto analysis = ReuseAnalysis::analyze(op);
    if (!analysis) {
      return failure();
    }

    auto sinks = op->getBlock()->getOps<SinkOp>();
    // We search `reverse(sinks)` rather than `sinks` because this tends
    // to give more readable results.
    auto reversedSinks = llvm::reverse(sinks);
    const auto reusableSink = llvm::find_if(
        reversedSinks, [&](SinkOp sink) { return analysis->canReuse(sink); });

    if (reusableSink == reversedSinks.end()) {
      return failure();
    }

    rewriteForReuse(op, *reusableSink, *analysis, rewriter);
    return success();
  }
};

/**
 * @brief This pass searches for qubits that do not interact with each other
 * directly or indirectly and attempts to reset and reuse one of them for the
 * other.
 */
struct ReuseQubits final : impl::ReuseQubitsBase<ReuseQubits> {
  using ReuseQubitsBase::ReuseQubitsBase;

protected:
  void runOnOperation() override {
    auto op = getOperation();
    auto* ctx = &getContext();

    // Define the set of patterns to use.
    RewritePatternSet patterns(ctx);
    patterns.add<ReuseQubitsPattern>(patterns.getContext());

    // Apply patterns in an iterative and greedy manner.
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco
