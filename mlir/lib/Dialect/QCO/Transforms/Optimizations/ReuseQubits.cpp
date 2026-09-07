/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cassert>
#include <cstddef>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_REUSEQUBITS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

/// Check an operation's own effects; recursive containers also need their
/// bodies.
static bool hasNoLocalEffects(Operation* op) {
  if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
    return effects.hasNoEffect();
  }
  return op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
}

/// Summarize each helper once, completing callers only after all callees.
/// Unknown targets and cycles never become proven effect-free.
static DenseSet<Operation*>
collectEffectFreeFunctions(ModuleOp moduleOp, SymbolTableCollection& symbols) {
  DenseMap<Operation*, size_t> pending;
  DenseMap<Operation*, SmallVector<Operation*>> callers;
  DenseSet<Operation*> blocked;
  moduleOp.walk([&](func::FuncOp function) {
    if (mqt::isUnitaryFunction(function) && !function.isExternal()) {
      pending.try_emplace(function, 0);
    }
  });
  for (auto& [function, count] : pending) {
    function->walk([&](Operation* op) {
      if (op == function) {
        return;
      }
      if (auto call = dyn_cast<CallOp>(op)) {
        auto callee = symbols.lookupNearestSymbolFrom<func::FuncOp>(
            call, call.getCalleeAttr());
        if (!callee || !pending.contains(callee)) {
          blocked.insert(function);
          return;
        }
        ++count;
        callers[callee].push_back(function);
      } else if (!hasNoLocalEffects(op)) {
        blocked.insert(function);
      }
    });
  }
  SmallVector<Operation*> ready;
  for (auto [function, count] : pending) {
    if (count == 0 && !blocked.contains(function)) {
      ready.push_back(function);
    }
  }
  DenseSet<Operation*> effectFree;
  while (!ready.empty()) {
    auto* function = ready.pop_back_val();
    effectFree.insert(function);
    for (auto* caller : callers[function]) {
      if (--pending[caller] == 0 && !blocked.contains(caller)) {
        ready.push_back(caller);
      }
    }
  }
  return effectFree;
}

static bool
isEffectFreeForReuse(Operation* root,
                     const DenseSet<Operation*>& effectFreeFunctions,
                     SymbolTableCollection& symbols) {
  SmallVector<Operation*> worklist{root};
  while (!worklist.empty()) {
    auto* op = worklist.pop_back_val();
    if (auto call = dyn_cast<CallOp>(op)) {
      auto callee = symbols.lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || !effectFreeFunctions.contains(callee)) {
        return false;
      }
    } else if (!hasNoLocalEffects(op)) {
      return false;
    }
    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      for (auto& region : op->getRegions()) {
        for (auto& block : region) {
          for (auto& nested : block) {
            worklist.push_back(&nested);
          }
        }
      }
    }
  }
  return true;
}

namespace {
class ReuseAnalysis {
public:
  [[nodiscard]] static std::optional<ReuseAnalysis>
  analyze(AllocOp alloc, const DenseSet<Operation*>& effectFreeFunctions,
          SymbolTableCollection& symbols) {
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
          isEffectFreeForReuse(&operation, effectFreeFunctions, symbols)) {
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
  ReuseQubitsPattern(MLIRContext* context,
                     const DenseSet<Operation*>& effectFreeFunctions,
                     SymbolTableCollection& symbols)
      : OpRewritePattern(context), effectFreeFunctions_(effectFreeFunctions),
        symbols_(symbols) {}

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

    const auto analysis =
        ReuseAnalysis::analyze(op, effectFreeFunctions_, symbols_);
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

private:
  const DenseSet<Operation*>& effectFreeFunctions_;
  SymbolTableCollection& symbols_;
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

    SymbolTableCollection symbols;
    const auto effectFreeFunctions = collectEffectFreeFunctions(op, symbols);

    /// Keep summarized helper bodies unchanged for this pass invocation.
    RewritePatternSet patterns(ctx);
    patterns.add<ReuseQubitsPattern>(ctx, effectFreeFunctions, symbols);
    const FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    auto result = op.walk<WalkOrder::PreOrder>([&](func::FuncOp function) {
      if (mqt::isUnitaryFunction(function) || function.isExternal()) {
        return WalkResult::skip();
      }
      /// Rewrite nested functions separately, without folding their helpers
      /// through an enclosing function's greedy driver.
      if (function
              .walk([&](func::FuncOp nested) {
                return nested == function ? WalkResult::advance()
                                          : WalkResult::interrupt();
              })
              .wasInterrupted()) {
        return WalkResult::advance();
      }
      if (failed(applyPatternsGreedily(function, frozenPatterns))) {
        return WalkResult::interrupt();
      }
      return WalkResult::skip();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco
