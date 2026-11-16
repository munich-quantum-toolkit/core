/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/IR/MQTAttributes.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <utility>

namespace mlir::mqt {
#define GEN_PASS_DEF_REMOVEUNUSEDQUBITS
#include "mqt/Dialect/MQT/Transforms/Passes.h.inc"

namespace {
template <typename ReleaseOp, typename AllocOp>
struct RemoveUnusedAllocation final : OpRewritePattern<ReleaseOp> {
  using OpRewritePattern<ReleaseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReleaseOp op,
                                PatternRewriter& rewriter) const override {
    auto value = op->getOperand(0);
    auto alloc = value.template getDefiningOp<AllocOp>();
    if (!alloc || !value.hasOneUse()) {
      return failure();
    }
    rewriter.eraseOp(op);
    rewriter.eraseOp(alloc);
    return success();
  }
};

struct RemoveUnusedQubits final
    : impl::RemoveUnusedQubitsBase<RemoveUnusedQubits> {
protected:
  void runOnOperation() override {
    if (preservesProgramQubits(getOperation())) {
      markAllAnalysesPreserved();
      return;
    }
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveUnusedAllocation<qc::DeallocOp, qc::AllocOp>,
                 RemoveUnusedAllocation<qco::SinkOp, qco::AllocOp>,
                 RemoveUnusedAllocation<qtensor::DeallocOp, qtensor::AllocOp>>(
        &getContext());
    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::mqt
