/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/IR/QCOps.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qc;

namespace {

/// pow(1.0) @ g  =>  g
struct InlinePow1 final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getExponentValue() != 1.0) {
      return failure();
    }
    auto* innerOp = op.getBodyUnitary().getOperation();
    rewriter.moveOpBefore(innerOp, op);
    rewriter.replaceOp(op, innerOp->getResults());
    return success();
  }
};

/// pow(0.0) @ g  =>  erase (identity / no-op)
struct ErasePow0 final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getExponentValue() != 0.0) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// pow(p) where p < 0  =>  pow(-p) { inv { g } }
struct NegPowToInvPow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getExponentValue() >= 0.0) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<PowOp>(op, -op.getExponentValue(), [&] {
      InvOp::create(rewriter, op.getLoc(), [&] {
        rewriter.clone(*op.getBodyUnitary().getOperation());
      });
    });
    return success();
  }
};

/// pow(a, pow(b, g))  =>  pow(a*b, g)
struct MergeNestedPow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto innerPow = llvm::dyn_cast<PowOp>(op.getBodyUnitary().getOperation());
    if (!innerPow) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<PowOp>(
        op, op.getExponentValue() * innerPow.getExponentValue(),
        [&] { rewriter.clone(*innerPow.getBodyUnitary().getOperation()); });
    return success();
  }
};

/// pow(p, ctrl(q, g))  =>  ctrl(q, pow(p, g))
struct MoveCtrlOutside final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto innerCtrl = llvm::dyn_cast<CtrlOp>(op.getBodyUnitary().getOperation());
    if (!innerCtrl) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<CtrlOp>(op, innerCtrl.getControls(), [&] {
      PowOp::create(rewriter, op.getLoc(), op.getExponentValue(), [&] {
        rewriter.clone(*innerCtrl.getBodyUnitary().getOperation());
      });
    });
    return success();
  }
};

} // namespace

UnitaryOpInterface PowOp::getBodyUnitary() {
  return llvm::cast<UnitaryOpInterface>(*(++getBody()->rbegin()));
}

void PowOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  double exponent,
                  const llvm::function_ref<void()>& bodyBuilder) {
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsState.addAttribute("exponent", odsBuilder.getF64FloatAttr(exponent));
  auto* region = odsState.addRegion();
  auto& block = region->emplaceBlock();

  odsBuilder.setInsertionPointToStart(&block);
  bodyBuilder();
  YieldOp::create(odsBuilder, odsState.location);
}

LogicalResult PowOp::verify() {
  auto& block = *getBody();
  if (block.getOperations().size() < 2) {
    return emitOpError("body region must have at least two operations");
  }
  if (!llvm::isa<YieldOp>(block.back())) {
    return emitOpError(
        "last operation in body region must be a yield operation");
  }
  auto iter = ++block.rbegin();
  if (!llvm::isa<UnitaryOpInterface>(*iter)) {
    return emitOpError(
        "second to last operation in body region must be a unitary operation");
  }
  for (auto it = ++iter; it != block.rend(); ++it) {
    if (llvm::isa<UnitaryOpInterface>(*it)) {
      return emitOpError("body region may only contain a single unitary op");
    }
  }
  return success();
}

void PowOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<InlinePow1, ErasePow0, NegPowToInvPow, MergeNestedPow,
              MoveCtrlOutside>(context);
}
