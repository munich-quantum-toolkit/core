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
#include "mlir/Dialect/QCO/Transforms/Decomposition/MultiControlled.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <mlir/Dialect/Arith/IR/Arith.h> // IWYU pragma: keep (Passes.h.inc)
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cstdint>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_DECOMPOSEMULTICONTROLLED
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

/**
 * @brief Decomposes a multi-controlled Pauli-X or Pauli-Z gate into elementary
 * one- and two-qubit gates.
 *
 * @details Matches `qco.ctrl` with a single `qco.x` or `qco.z` body when the
 * control count is at least `minControls_` (and at least two, as enforced by
 * the pass). Single-control `CX`/`CZ` and other gates are left unchanged.
 */
struct DecomposeMultiControlledPauliPattern final : OpRewritePattern<CtrlOp> {
  explicit DecomposeMultiControlledPauliPattern(MLIRContext* context,
                                                uint64_t minControls)
      : OpRewritePattern<CtrlOp>(context), minControls_(minControls) {}

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumControls() < minControls_ || op.getNumTargets() != 1) {
      return failure();
    }
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }

    rewriter.setInsertionPoint(op);
    const auto controls = op.getControlsIn();
    const auto target = op.getInputTarget(0);
    const auto loc = op.getLoc();

    if (isa<XOp>(inner.getOperation())) {
      rewriter.replaceOp(
          op, decomposition::synthesizeMcx(rewriter, loc, controls, target));
      return success();
    }
    if (isa<ZOp>(inner.getOperation())) {
      rewriter.replaceOp(
          op, decomposition::synthesizeMcz(rewriter, loc, controls, target));
      return success();
    }
    return failure();
  }

private:
  uint64_t minControls_;
};

/**
 * @brief Pass that decomposes multi-controlled X and Z gates into elementary
 * gates.
 */
struct DecomposeMultiControlled final
    : impl::DecomposeMultiControlledBase<DecomposeMultiControlled> {
  using DecomposeMultiControlledBase::DecomposeMultiControlledBase;

protected:
  void runOnOperation() override {
    if (minControls < 2) {
      getOperation().emitError()
          << "decompose-multi-controlled requires min-controls >= 2";
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeMultiControlledPauliPattern>(&getContext(),
                                                       minControls);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco
