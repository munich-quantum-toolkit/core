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
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_REMOVEDEADGATES
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

/**
 * @brief Remove dead quantum operations that feed a sink.
 */
struct RemoveDeadGatesBeforeSink final : OpRewritePattern<SinkOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SinkOp op,
                                PatternRewriter& rewriter) const override {
    return tryEliminateDeadGateValue(op.getQubit(), rewriter);
  }
};

/**
 * @brief Remove dead quantum operations that precede a reset.
 */
struct RemoveDeadGatesBeforeReset final : OpRewritePattern<ResetOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ResetOp op,
                                PatternRewriter& rewriter) const override {
    return tryEliminateDeadGateValue(op.getQubitIn(), rewriter);
  }
};

struct RemoveDeadGates final : impl::RemoveDeadGatesBase<RemoveDeadGates> {
  using RemoveDeadGatesBase::RemoveDeadGatesBase;

protected:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveDeadGatesBeforeSink, RemoveDeadGatesBeforeReset>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco
