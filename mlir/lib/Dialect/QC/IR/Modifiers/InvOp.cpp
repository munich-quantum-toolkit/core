/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/IR/QCDialect.h"

#include <cstddef>
#include <functional>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qc;

namespace {

/**
 * @brief Move nested control modifiers outside, i.e., `inv(ctrl(x)) =>
 * ctrl(inv(x))`.
 */
struct MoveCtrlOutside final : OpRewritePattern<InvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InvOp invOp,
                                PatternRewriter& rewriter) const override {
    auto bodyUnitary = invOp.getBodyUnitary();
    auto innerCtrlOp = llvm::dyn_cast<CtrlOp>(bodyUnitary.getOperation());
    if (!innerCtrlOp) {
      return failure();
    }

    auto controls = innerCtrlOp.getControls();
    rewriter.replaceOpWithNewOp<CtrlOp>(invOp, controls, [&] {
      InvOp::create(rewriter, invOp.getLoc(), [&] {
        rewriter.clone(*innerCtrlOp.getBodyUnitary().getOperation());
      });
    });

    return success();
  }
};

/**
 * @brief Remove inverse modifiers around self-adjoint gates.
 *
 * For self-adjoint gates U (i.e., U = U†), inv(U) = U holds.
 */
struct InlineSelfAdjoint final : OpRewritePattern<InvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InvOp op,
                                PatternRewriter& rewriter) const override {
    auto* innerOp = op.getBodyUnitary().getOperation();

    if (!llvm::isa<IdOp, HOp, XOp, YOp, ZOp, SWAPOp, BarrierOp>(innerOp)) {
      return failure();
    }

    rewriter.moveOpBefore(innerOp, op);
    rewriter.replaceOp(op, innerOp->getResults());
    return success();
  }
};

/**
 * @brief Replace inverse modifiers around gates where the inverse is a known
 * gate by their known inverse.
 *
 * For example, for the T gate, inv(T) = Tdg holds.
 */
struct ReplaceWithKnownGates final : OpRewritePattern<InvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InvOp op,
                                PatternRewriter& rewriter) const override {
    auto* innerOp = op.getBodyUnitary().getOperation();

    return llvm::TypeSwitch<Operation*, LogicalResult>(innerOp)
        .Case<GPhaseOp>([&](auto g) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), g.getTheta());
          rewriter.replaceOpWithNewOp<GPhaseOp>(op, negTheta);
          return success();
        })
        .Case<TOp>([&](auto) {
          rewriter.replaceOpWithNewOp<TdgOp>(op, op.getTarget(0));
          return success();
        })
        .Case<TdgOp>([&](auto) {
          rewriter.replaceOpWithNewOp<TOp>(op, op.getTarget(0));
          return success();
        })
        .Case<SOp>([&](auto) {
          rewriter.replaceOpWithNewOp<SdgOp>(op, op.getTarget(0));
          return success();
        })
        .Case<SdgOp>([&](auto) {
          rewriter.replaceOpWithNewOp<SOp>(op, op.getTarget(0));
          return success();
        })
        .Case<SXOp>([&](auto) {
          rewriter.replaceOpWithNewOp<SXdgOp>(op, op.getTarget(0));
          return success();
        })
        .Case<SXdgOp>([&](auto) {
          rewriter.replaceOpWithNewOp<SXOp>(op, op.getTarget(0));
          return success();
        })
        .Case<POp>([&](auto p) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), p.getTheta());
          rewriter.replaceOpWithNewOp<POp>(op, op.getTarget(0), negTheta);
          return success();
        })
        .Case<ROp>([&](auto r) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), r.getTheta());
          rewriter.replaceOpWithNewOp<ROp>(op, op.getTarget(0), negTheta,
                                           r.getPhi());
          return success();
        })
        .Case<RXOp>([&](auto rx) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), rx.getTheta());
          rewriter.replaceOpWithNewOp<RXOp>(op, op.getTarget(0), negTheta);
          return success();
        })
        .Case<UOp>([&](auto u) {
          Value newPhi =
              arith::NegFOp::create(rewriter, op.getLoc(), u.getLambda());
          Value newLambda =
              arith::NegFOp::create(rewriter, op.getLoc(), u.getPhi());
          Value newTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), u.getTheta());
          rewriter.replaceOpWithNewOp<UOp>(op, op.getTarget(0), newTheta,
                                           newPhi, newLambda);
          return success();
        })
        .Case<U2Op>([&](auto u) {
          auto pi = arith::ConstantOp::create(
              rewriter, op.getLoc(),
              rewriter.getF64FloatAttr(std::numbers::pi));
          Value newPhi =
              arith::NegFOp::create(rewriter, op.getLoc(), u.getLambda());
          newPhi = arith::SubFOp::create(rewriter, op.getLoc(), newPhi, pi);
          Value newLambda =
              arith::NegFOp::create(rewriter, op.getLoc(), u.getPhi());
          newLambda =
              arith::AddFOp::create(rewriter, op.getLoc(), newLambda, pi);
          rewriter.replaceOpWithNewOp<U2Op>(op, op.getTarget(0), newPhi,
                                            newLambda);
          return success();
        })
        .Case<DCXOp>([&](auto) {
          rewriter.replaceOpWithNewOp<DCXOp>(op, op.getTarget(1),
                                             op.getTarget(0));
          return success();
        })
        .Case<RXXOp>([&](auto rxx) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), rxx.getTheta());
          rewriter.replaceOpWithNewOp<RXXOp>(op, op.getTarget(0),
                                             op.getTarget(1), negTheta);
          return success();
        })
        .Case<RYOp>([&](auto ry) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), ry.getTheta());
          rewriter.replaceOpWithNewOp<RYOp>(op, op.getTarget(0), negTheta);
          return success();
        })
        .Case<RYYOp>([&](auto ryy) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), ryy.getTheta());
          rewriter.replaceOpWithNewOp<RYYOp>(op, op.getTarget(0),
                                             op.getTarget(1), negTheta);
          return success();
        })
        .Case<RZOp>([&](auto rz) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), rz.getTheta());
          rewriter.replaceOpWithNewOp<RZOp>(op, op.getTarget(0), negTheta);
          return success();
        })
        .Case<RZXOp>([&](auto rzx) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), rzx.getTheta());
          rewriter.replaceOpWithNewOp<RZXOp>(op, op.getTarget(0),
                                             op.getTarget(1), negTheta);
          return success();
        })
        .Case<RZZOp>([&](auto rzz) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), rzz.getTheta());
          rewriter.replaceOpWithNewOp<RZZOp>(op, op.getTarget(0),
                                             op.getTarget(1), negTheta);
          return success();
        })
        .Case<XXMinusYYOp>([&](auto xxminusyy) {
          Value negTheta = arith::NegFOp::create(rewriter, op.getLoc(),
                                                 xxminusyy.getTheta());
          rewriter.replaceOpWithNewOp<XXMinusYYOp>(op, op.getTarget(0),
                                                   op.getTarget(1), negTheta,
                                                   xxminusyy.getBeta());
          return success();
        })
        .Case<XXPlusYYOp>([&](auto xxplusyy) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), xxplusyy.getTheta());
          rewriter.replaceOpWithNewOp<XXPlusYYOp>(op, op.getTarget(0),
                                                  op.getTarget(1), negTheta,
                                                  xxplusyy.getBeta());
          return success();
        })
        .Default([&](auto) { return failure(); });
  }
};

/**
 * @brief Cancel nested inverse modifiers, i.e., `inv(inv(x)) => x`.
 */
struct CancelNestedInv final : OpRewritePattern<InvOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(InvOp invOp,
                                PatternRewriter& rewriter) const override {
    auto innerUnitary = invOp.getBodyUnitary();
    auto innerInvOp = llvm::dyn_cast<InvOp>(innerUnitary.getOperation());
    if (!innerInvOp) {
      return failure();
    }

    auto innerInnerUnitary = innerInvOp.getBodyUnitary().getOperation();
    rewriter.moveOpBefore(innerInnerUnitary, invOp);
    rewriter.replaceOp(invOp, innerInnerUnitary->getResults());

    return success();
  }
};

} // namespace

UnitaryOpInterface InvOp::getBodyUnitary() {
  // In principle, the body region should only contain exactly two operations,
  // the actual unitary operation and a yield operation. However, the region may
  // also contain constants and arithmetic operations, e.g., created as part of
  // canonicalization. Thus, the only safe way to access the unitary operation
  // is to get the second operation from the back of the region.
  return llvm::dyn_cast<UnitaryOpInterface>(*(++getBody()->rbegin()));
}

size_t InvOp::getNumQubits() { return getBodyUnitary().getNumQubits(); }

size_t InvOp::getNumTargets() { return getNumQubits(); }

size_t InvOp::getNumControls() { return 0; }

Value InvOp::getQubit(const size_t i) { return getBodyUnitary().getQubit(i); }

Value InvOp::getTarget(const size_t i) { return getQubit(i); }

Value InvOp::getControl([[maybe_unused]] const size_t i) {
  llvm::reportFatalUsageError("Operation does not have controls");
}

size_t InvOp::getNumParams() { return getBodyUnitary().getNumParams(); }

Value InvOp::getParameter(const size_t i) {
  return getBodyUnitary().getParameter(i);
}

void InvOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  UnitaryOpInterface bodyUnitary) {
  const OpBuilder::InsertionGuard guard(odsBuilder);
  auto* region = odsState.addRegion();
  auto& block = region->emplaceBlock();

  // Move the unitary op into the block
  odsBuilder.setInsertionPointToStart(&block);
  odsBuilder.clone(*bodyUnitary.getOperation());
  YieldOp::create(odsBuilder, odsState.location);
}

void InvOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  const llvm::function_ref<void()>& bodyBuilder) {
  const OpBuilder::InsertionGuard guard(odsBuilder);
  auto* region = odsState.addRegion();
  auto& block = region->emplaceBlock();

  odsBuilder.setInsertionPointToStart(&block);
  bodyBuilder();
  YieldOp::create(odsBuilder, odsState.location);
}

LogicalResult InvOp::verify() {
  auto& block = *getBody();
  if (block.getOperations().size() < 2) {
    return emitOpError("body region must have at least two operations");
  }
  if (!llvm::isa<YieldOp>(block.back())) {
    return emitOpError(
        "last operation in body region must be a yield operation");
  }
  auto iter = ++block.rbegin();
  if (!llvm::isa<UnitaryOpInterface>(*(iter))) {
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

void InvOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<CancelNestedInv, MoveCtrlOutside, InlineSelfAdjoint,
              ReplaceWithKnownGates>(context);
}
