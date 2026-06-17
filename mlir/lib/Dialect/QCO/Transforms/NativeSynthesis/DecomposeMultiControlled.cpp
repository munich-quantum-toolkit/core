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
#include "mlir/Dialect/QCO/Transforms/Decomposition/MultiControlled.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <mlir/Dialect/Arith/IR/Arith.h> // IWYU pragma: keep (Passes.h.inc)
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::qco {

#define GEN_PASS_DEF_DECOMPOSEMULTICONTROLLED
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

/**
 * @brief Decomposes a multi-controlled X gate into elementary one- and
 * two-qubit gates.
 */
struct DecomposeMultiControlledXPattern final : OpRewritePattern<CtrlOp> {
  explicit DecomposeMultiControlledXPattern(MLIRContext* context,
                                            uint64_t minControls)
      : OpRewritePattern<CtrlOp>(context), minControls_(minControls) {}

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumControls() < minControls_) {
      return failure();
    }
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner || !isa<XOp>(inner.getOperation()) || op.getNumTargets() != 1) {
      return failure();
    }

    rewriter.setInsertionPoint(op);
    const auto results = decomposition::synthesizeMcx(
        rewriter, op.getLoc(), op.getControlsIn(), op.getInputTarget(0));
    rewriter.replaceOp(op, results);
    return success();
  }

private:
  uint64_t minControls_;
};

/**
 * @brief Pass that decomposes multi-controlled X gates into elementary gates.
 */
struct DecomposeMultiControlled final
    : impl::DecomposeMultiControlledBase<DecomposeMultiControlled> {
  using DecomposeMultiControlledBase::DecomposeMultiControlledBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeMultiControlledXPattern>(&getContext(), minControls);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco
