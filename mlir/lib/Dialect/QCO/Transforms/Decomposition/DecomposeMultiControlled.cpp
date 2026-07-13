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
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_DECOMPOSEMULTICONTROLLED
#define GEN_PASS_DEF_DECOMPOSETHREECONTROLLED
#define GEN_PASS_DEF_DECOMPOSETWOCONTROLLED
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {
struct ControlledGateSpec {
  decomposition::ControlledTarget gate;
  std::optional<double> theta;
};
} // namespace

static std::optional<ControlledGateSpec>
matchControlledGate(UnitaryOpInterface inner) {
  if (isa<XOp>(inner.getOperation())) {
    return ControlledGateSpec{.gate = decomposition::ControlledTarget::X,
                              .theta = std::nullopt};
  }
  if (isa<ZOp>(inner.getOperation())) {
    return ControlledGateSpec{.gate = decomposition::ControlledTarget::Z,
                              .theta = std::nullopt};
  }
  if (auto pOp = dyn_cast<POp>(inner.getOperation())) {
    if (const auto theta = utils::valueToDouble(pOp.getTheta())) {
      return ControlledGateSpec{.gate = decomposition::ControlledTarget::Phase,
                                .theta = theta};
    }
  }
  return std::nullopt;
}

namespace {
struct DecomposeTwoControlledPattern final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumControls() != 2 || op.getNumTargets() != 1) {
      return failure();
    }
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    const auto spec = matchControlledGate(inner);
    if (!spec) {
      return failure();
    }

    rewriter.setInsertionPoint(op);
    rewriter.replaceOp(op, decomposition::synthesizeTwoControlled(
                               rewriter, op.getLoc(), op.getControlsIn()[0],
                               op.getControlsIn()[1], op.getInputTarget(0),
                               spec->gate, spec->theta));
    return success();
  }
};

struct DecomposeRCCXPattern final : OpRewritePattern<RCCXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RCCXOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.setInsertionPoint(op);
    rewriter.replaceOp(op, decomposition::synthesizeRCCX(
                               rewriter, op.getLoc(), op.getInputQubit(0),
                               op.getInputQubit(1), op.getInputQubit(2)));
    return success();
  }
};

struct DecomposeTwoControlled final
    : impl::DecomposeTwoControlledBase<DecomposeTwoControlled> {
  using DecomposeTwoControlledBase::DecomposeTwoControlledBase;

protected:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeTwoControlledPattern, DecomposeRCCXPattern>(
        &getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct DecomposeThreeControlledPattern final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumControls() != 3 || op.getNumTargets() != 1) {
      return failure();
    }
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    const auto spec = matchControlledGate(inner);
    if (!spec) {
      return failure();
    }

    rewriter.setInsertionPoint(op);
    rewriter.replaceOp(op, decomposition::synthesizeThreeControlled(
                               rewriter, op.getLoc(), op.getControlsIn(),
                               op.getInputTarget(0), spec->gate, spec->theta));
    return success();
  }
};

struct DecomposeThreeControlled final
    : impl::DecomposeThreeControlledBase<DecomposeThreeControlled> {
  using DecomposeThreeControlledBase::DecomposeThreeControlledBase;

protected:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeThreeControlledPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct DecomposeMultiControlledPauliPattern final : OpRewritePattern<CtrlOp> {
  explicit DecomposeMultiControlledPauliPattern(MLIRContext* context,
                                                uint64_t minControls)
      : OpRewritePattern<CtrlOp>(context), minControls_(minControls) {}

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    const auto numControls = op.getNumControls();
    if (numControls < minControls_ || numControls < 4 ||
        op.getNumTargets() != 1) {
      return failure();
    }
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    const auto spec = matchControlledGate(inner);
    if (!spec) {
      return failure();
    }
    // HP24 core only supports Pauli-X/Z; leave k >= 4 phase gates untouched.
    if (spec->gate == decomposition::ControlledTarget::Phase) {
      return failure();
    }

    rewriter.setInsertionPoint(op);
    rewriter.replaceOp(op, decomposition::synthesizeMultiControlled(
                               rewriter, op.getLoc(), op.getControlsIn(),
                               op.getInputTarget(0), minControls_, spec->gate,
                               spec->theta));
    return success();
  }

private:
  uint64_t minControls_;
};

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
