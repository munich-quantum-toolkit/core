/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_COMBINECONTROLLEDGATES
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

struct CombineControls final : OpRewritePattern<CtrlOp> {
  CombineControls(MLIRContext* context, const CompilerTarget* target)
      : OpRewritePattern(context), target_(target) {}

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumControls() == 0) {
      return failure();
    }
    auto next = dyn_cast<CtrlOp>(*op.getOutputControl(0).user_begin());
    if (!next || next->getBlock() != op->getBlock() ||
        op.getNumControls() != next.getNumControls() ||
        !llvm::equal(op.getOutputQubits(), next.getInputQubits())) {
      return failure();
    }
    if (target_ != nullptr) {
      if (target_->supports(next)) {
        return failure();
      }
      for (auto unitary = cast<UnitaryOpInterface>(op.getOperation()); unitary;
           unitary = unitary->getParentOfType<UnitaryOpInterface>()) {
        if (target_->supports(unitary)) {
          return failure();
        }
      }
    }

    // Linear wires bypass intervening operations; moving the shell preserves
    // parameter dominance and appends each body only once in long chains.
    rewriter.moveOpBefore(op, next);
    rewriter.modifyOpInPlace(op, [&] {
      auto* yield = op.getBody()->getTerminator();
      SmallVector<Value> values(yield->getOperands());
      rewriter.inlineBlockBefore(next.getBody(), yield, values);
      rewriter.eraseOp(yield);
    });
    rewriter.replaceOp(next, op.getResults());
    return success();
  }

private:
  const CompilerTarget* target_;
};

struct CombineControlledGates final
    : impl::CombineControlledGatesBase<CombineControlledGates> {
  using Base::Base;

  CombineControlledGates() = default;

  explicit CombineControlledGates(const CompilerTarget& target)
      : target_(target) {}

protected:
  void runOnOperation() override {
    const auto* target = target_ ? &*target_ : nullptr;
    if (useTarget) {
      const auto& environment = getAnalysis<TargetEnvironmentAnalysis>();
      if (!environment) {
        getOperation().emitError()
            << "expected a valid mqt.target_env: " << environment.error();
        signalPassFailure();
        return;
      }
      target = &environment.environment().target();
    }
    RewritePatternSet patterns(&getContext());
    patterns.add<CombineControls>(&getContext(), target);
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns),
            GreedyRewriteConfig{}.setUseTopDownTraversal()))) {
      signalPassFailure();
    }
  }

private:
  std::optional<CompilerTarget> target_;
};

} // namespace

std::unique_ptr<Pass>
createCombineControlledGates(const CompilerTarget& target) {
  return std::make_unique<CombineControlledGates>(target);
}

} // namespace mlir::qco
